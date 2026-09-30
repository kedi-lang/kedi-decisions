"""One local decision model, reusable by Pydantic AI without a hosted API key."""

from typing import Annotated, Literal

from kedi_laya import ChoiceCriteria, LayaClient, Probability
from pydantic import BaseModel
from pydantic_ai import Agent


class Ticket(BaseModel):
    team: Annotated[
        Literal["billing", "technical"],
        ChoiceCriteria({"billing": "Invoices and charges", "technical": "Broken software"}),
    ]
    refund_requested: Annotated[Probability, "Does the customer request a refund?"]


if __name__ == "__main__":
    with LayaClient("aac6fef/laya-multilingual-mlx") as client:
        agent = Agent(client.as_pydantic_model(), output_type=Ticket)
        print(agent.run_sync("I was charged twice. Please refund the duplicate payment.").output)
