"""Provider-independent contracts exercised through real framework runtimes."""

import subprocess
import sys

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters

from kedi_decisions import DecisionEvaluator
from kedi_decisions.contracts import (
    BooleanAnswer,
    BooleanQuestion,
    ChoiceAnswer,
    DecisionResponse,
    DecisionUsage,
)
from kedi_decisions.integrations.langchain import DecisionChatModel
from kedi_decisions.integrations.pydantic import DecisionModel
from kedi_decisions.routing import ToolCallProposed


class Review(BaseModel):
    approved: bool


class Client:
    def __init__(self):
        self.requests = []

    def decide(self, state, questions, *, model=None):
        self.requests.append((state, questions))
        answers = {}
        for name, question in questions.items():
            if isinstance(question, BooleanQuestion):
                answers[name] = BooleanAnswer(probability=0.9)
            else:
                selected = list(question.criteria)[-1]
                answers[name] = ChoiceAnswer(
                    choice=selected,
                    confidence=0.95,
                    probabilities={key: float(key == selected) for key in question.criteria},
                )
        return DecisionResponse(
            model=model, answers=answers, usage=DecisionUsage(input_tokens=18, output_tokens=0)
        )


class AsyncClient:
    def __init__(self, client):
        self.client = client

    async def decide(self, state, questions, *, model=None):
        return self.client.decide(state, questions, model=model)


@pytest.fixture
def evaluator():
    client = Client()
    return DecisionEvaluator(
        "test-decision", sync_client=client, client=AsyncClient(client), provider_name="local-test"
    )


def test_core_and_laya_imports_do_not_import_frameworks_or_vendor_sdks():
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import sys
import kedi_decisions
import kedi_laya
for forbidden in ('typesafe_sdk', 'kedi_typesafe', 'pydantic_ai', 'langchain', 'langchain_core', 'laya', 'laya_mlx', 'mlx', 'torch'):
    assert forbidden not in sys.modules, forbidden
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_pydantic_output_modes_threshold_metadata_and_usage(evaluator, native, stream):
    model = DecisionModel(evaluator)
    agent = Agent(model, output_type=NativeOutput(Review) if native else Review)
    settings = {"decision_threshold": 0.95}
    if stream:
        async with agent.run_stream("Review this request", model_settings=settings) as result:
            output = await result.get_output()
            assert output.approved is False
            assert result.usage.input_tokens == 18
            messages = result.new_messages()
    else:
        result = await agent.run("Review this request", model_settings=settings)
        assert result.output.approved is False
        assert result.usage.input_tokens == 18
        messages = result.new_messages()
    response = next(message for message in reversed(messages) if isinstance(message, ModelResponse))
    assert response.provider_details["decisions"]["provider"] == "local-test"
    assert response.provider_details["decisions"]["boolean_threshold"] == 0.95


@pytest.mark.asyncio
async def test_zero_argument_tool_executes_once_before_structured_output(evaluator):
    calls = []

    def lookup_policy() -> str:
        """Fetch the policy needed for this review."""
        calls.append("lookup")
        return "The request is approved."

    result = await Agent(DecisionModel(evaluator), tools=[lookup_policy], output_type=Review).run(
        "Review using the policy"
    )
    assert result.output.approved is True
    assert calls == ["lookup"]
    assert result.usage.requests == 2
    assert "lookup_policy" in str(evaluator._sync_client.requests[-1][0])


@pytest.mark.asyncio
async def test_parameterized_tool_requires_argument_producer(evaluator):
    def lookup_policy(policy_id: str) -> str:
        raise AssertionError("The model must not invent a policy id")

    with pytest.raises(ToolCallProposed, match="argument-producing"):
        await Agent(DecisionModel(evaluator), tools=[lookup_policy], output_type=Review).run(
            "Review using the policy"
        )


@pytest.mark.asyncio
async def test_freeform_and_unsupported_settings_fail_before_inference(evaluator):
    with pytest.raises(UserError, match="structured output"):
        await Agent(DecisionModel(evaluator)).run("Write some prose")
    with pytest.raises(UserError, match="temperature"):
        await Agent(DecisionModel(evaluator), output_type=Review).run(
            "Review", model_settings={"temperature": 0.2}
        )
    assert not evaluator._sync_client.requests


@pytest.mark.asyncio
async def test_output_handoff_and_threshold_no_selection(evaluator):
    def approve() -> bool:
        return True

    agent = Agent(DecisionModel(evaluator), output_type=[approve])
    assert (await agent.run("Approve")).output is True
    with pytest.raises(UserError, match="above the threshold"):
        # The synthetic client picks a handoff with probability 1; replace that
        # response with lower confidence to exercise the routing threshold.
        client = evaluator._sync_client
        original = client.decide

        def uncertain(*args, **kwargs):
            response = original(*args, **kwargs)
            from dataclasses import replace

            return replace(
                response,
                answers={
                    key: replace(
                        answer,
                        probabilities={
                            option: 1 / len(answer.probabilities) for option in answer.probabilities
                        },
                    )
                    for key, answer in response.answers.items()
                },
            )

        client.decide = uncertain

        def reject() -> bool:
            return False

        await Agent(DecisionModel(evaluator), output_type=[approve, reject]).run("Choose")


@pytest.mark.asyncio
async def test_langchain_threshold_sync_async_and_metadata(evaluator):
    model = DecisionChatModel("test-decision", evaluator=evaluator, decision_threshold=0.95)
    chain = model.with_structured_output(Review, include_raw=True)
    sync = chain.invoke("Review")
    asynchronous = await chain.ainvoke("Review")
    for result in (sync, asynchronous):
        assert result["parsed"].approved is False
        assert result["raw"].usage_metadata["input_tokens"] == 18
        assert result["raw"].response_metadata["decisions"]["provider"] == "local-test"
    assert model._identifying_params["threshold"] == 0.95


def test_missing_clients_and_invalid_threshold():
    with pytest.raises(ValueError, match="finite"):
        DecisionEvaluator("x", threshold=float("nan"))
    evaluator = DecisionEvaluator("x")
    with pytest.raises(ValueError, match="synchronous decision client"):
        evaluator.evaluate_sync(state="Review", schema=Review.model_json_schema())


@pytest.mark.parametrize("threshold", [-0.1, 1.1, True, "0.5"])
def test_invalid_core_thresholds(threshold):
    with pytest.raises(ValueError):
        DecisionEvaluator("test", threshold=threshold)


def test_empty_model_name_is_rejected():
    with pytest.raises(ValueError, match="model name"):
        DecisionEvaluator(" ")


@pytest.mark.asyncio
async def test_optional_extraction_without_candidates_never_calls_provider(evaluator):
    from typing import Annotated

    from pydantic import Field

    class Extract(BaseModel):
        reference: Annotated[str, Field(pattern=r"CASE-\d+")] | None

    schema = Extract.model_json_schema()
    for result in (
        evaluator.evaluate_sync(state="No reference here", schema=schema),
        await evaluator.evaluate(state="No reference here", schema=schema),
    ):
        assert result.values == {"reference": None}
        assert result.input_tokens == 0 and result.output_tokens == 0
    assert not evaluator._sync_client.requests
    await evaluator.aclose_current()
    await evaluator.aclose()


@pytest.mark.asyncio
async def test_pydantic_instructions_and_message_history_are_preserved(evaluator):
    from pydantic_ai.messages import RetryPromptPart, SystemPromptPart, TextPart

    from kedi_decisions.integrations.pydantic import messages_to_state

    await Agent(DecisionModel(evaluator), output_type=Review, instructions="Use the policy").run(
        "Review"
    )
    assert "Use the policy" in evaluator._sync_client.requests[-1][0]["instructions"]
    messages = [
        ModelRequest(parts=[SystemPromptPart("Policy"), UserPromptPart("First")]),
        ModelResponse(parts=[TextPart("Earlier response")]),
        ModelRequest(parts=[RetryPromptPart("Try again"), UserPromptPart("Second")]),
    ]
    state = messages_to_state(messages)
    assert state["text"] == "Second"
    assert [item["role"] for item in state["history"]] == ["system", "user", "assistant", "retry"]
    with pytest.raises(UserError, match="input to evaluate"):
        messages_to_state([])
    with pytest.raises(UserError, match="text-only"):
        messages_to_state([ModelRequest(parts=[UserPromptPart(["Not a plain text prompt"])])])


@pytest.mark.asyncio
async def test_async_client_is_required():
    evaluator = DecisionEvaluator("x")
    with pytest.raises(ValueError, match="async decision client"):
        await evaluator.evaluate(state="Review", schema=Review.model_json_schema())


@pytest.mark.asyncio
async def test_withheld_tools_are_not_sent_to_provider(evaluator):
    from pydantic_ai.tools import ToolDefinition

    parameters = ModelRequestParameters(
        output_mode="tool",
        function_tools=[ToolDefinition(name="hidden", defer_loading=True)],
        output_tools=[
            ToolDefinition(name="result", parameters_json_schema=Review.model_json_schema())
        ],
        allow_text_output=False,
    )
    response = await DecisionModel(evaluator).request(
        [ModelRequest(parts=[UserPromptPart("Review")])],
        None,
        parameters,
    )
    assert isinstance(response.parts[0], ToolCallPart)
    assert response.parts[0].tool_name == "result"
    assert set(evaluator._sync_client.requests[-1][1]) == {"approved"}
