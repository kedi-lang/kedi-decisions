"""Local inference checks. No hosted service, key, or model-generated text."""

import argparse
import asyncio
import json
import statistics
import time
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Literal

from kedi_laya import ChoiceCriteria, LayaClient, Probability, Rubric
from pydantic import BaseModel
from pydantic_ai import Agent


class Triage(BaseModel):
    department: Annotated[
        Literal["billing", "technical"],
        "The team that should handle the request",
        ChoiceCriteria(
            {
                "billing": "Payments, charges and invoices",
                "technical": "Broken software and unavailable services",
            }
        ),
    ]
    outage: Annotated[bool, "Is the service currently completely unavailable?"]
    outage_probability: Annotated[Probability, "Is the service currently completely unavailable?"]
    impact: Annotated[
        float,
        Rubric(["No service impact", "Partial service disruption", "Complete service outage"]),
    ]


async def measure(checkpoint: str, repeats: int) -> dict:
    cases = [
        (
            "I was charged twice for my subscription. Everything works. Please refund the duplicate charge.",
            "billing",
            False,
        ),
        (
            "Our application is completely down. Nobody can sign in or use any features. Please fix the server.",
            "technical",
            True,
        ),
        ("Please send an invoice for last month. The service works normally.", "billing", False),
        (
            "The export button is broken but the rest of the application works normally.",
            "technical",
            False,
        ),
    ]
    rows = []
    with LayaClient(checkpoint) as client:
        evaluator = client.as_evaluator()
        schema = Triage.model_json_schema()
        start = time.perf_counter()
        cold = await evaluator.evaluate(state=cases[0][0], schema=schema)
        cold_seconds = time.perf_counter() - start
        agent = Agent(client.as_pydantic_model(), output_type=Triage)
        langchain = client.as_langchain_model().with_structured_output(Triage)
        for repeat in range(repeats):
            for index, (state, department, outage) in enumerate(cases):
                for surface in ("decision-api", "pydantic", "langchain"):
                    start = time.perf_counter()
                    if surface == "decision-api":
                        result = await evaluator.evaluate(state=state, schema=schema)
                        value = Triage.model_validate(result.values)
                    elif surface == "pydantic":
                        value = (await agent.run(state)).output
                    else:
                        value = await langchain.ainvoke(state)
                    rows.append(
                        {
                            "surface": surface,
                            "case": index,
                            "repeat": repeat,
                            "seconds": time.perf_counter() - start,
                            "output": value.model_dump(),
                            "department_correct": value.department == department,
                            "outage_correct": value.outage == outage,
                        }
                    )
        overflow_rejected = False
        try:
            await evaluator.evaluate(state="long context " * 600, schema=schema)
        except ValueError as error:
            overflow_rejected = "context window" in str(error)
        return {
            "checkpoint": Path(checkpoint).name,
            "backend": "laya-mlx",
            "versions": {
                name: version(name)
                for name in (
                    "laya-mlx",
                    "mlx",
                    "kedi-decisions",
                    "kedi-laya",
                    "pydantic-ai-slim",
                    "langchain",
                )
            },
            "cold_seconds": cold_seconds,
            "cold_output": cold.values,
            "overflow_rejected": overflow_rejected,
            "summary": {
                surface: {
                    "calls": len(selected := [r for r in rows if r["surface"] == surface]),
                    "median_seconds": statistics.median(r["seconds"] for r in selected),
                    "max_seconds": max(r["seconds"] for r in selected),
                    "department_correct": sum(r["department_correct"] for r in selected),
                    "outage_correct": sum(r["outage_correct"] for r in selected),
                }
                for surface in ("decision-api", "pydantic", "langchain")
            },
            "rows": rows,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 20:
        parser.error("--repeats must be between 1 and 20")
    report = asyncio.run(measure(args.checkpoint, args.repeats))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
