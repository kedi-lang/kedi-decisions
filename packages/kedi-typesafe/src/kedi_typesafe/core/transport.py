"""Translate the shared contract at the TypeSafe SDK boundary."""

from kedi_decisions.contracts import (
    Answer,
    BooleanAnswer,
    BooleanQuestion,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionResponse,
    DecisionUsage,
    Question,
    ScoreAnswer,
)
from typesafe_sdk import Choice, Noul, NoulAnswer, NoulCriteria, Score, SystemOneResponse
from typesafe_sdk import ChoiceAnswer as SDKChoiceAnswer


def question_to_sdk(question: Question) -> Noul | Choice | Score:
    if isinstance(question, BooleanQuestion):
        return Noul(
            instructions=question.instructions,
            criteria=NoulCriteria(
                true=question.criteria.get("true"), false=question.criteria.get("false")
            ),
        )
    if isinstance(question, ChoiceQuestion):
        return Choice(instructions=question.instructions, criteria=dict(question.criteria))
    return Score(instructions=question.instructions, criteria=list(question.criteria))


def response_from_sdk(response: SystemOneResponse) -> DecisionResponse:
    answers: dict[str, Answer] = {}
    for name, answer in response.answers.items():
        if isinstance(answer, NoulAnswer):
            answers[name] = BooleanAnswer(probability=answer.noul)
        elif isinstance(answer, SDKChoiceAnswer):
            answers[name] = ChoiceAnswer(
                choice=answer.choice,
                confidence=answer.confidence,
                probabilities=answer.probabilities,
            )
        else:
            answers[name] = ScoreAnswer(
                score=answer.score,
                confidence=answer.confidence,
                probabilities=answer.probabilities,
                legend=answer.legend,
            )
    return DecisionResponse(
        model=response.model,
        answers=answers,
        usage=DecisionUsage(
            input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens
        ),
    )
