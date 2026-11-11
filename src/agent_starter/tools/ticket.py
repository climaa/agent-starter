"""Example tool: create a support ticket by POSTing to a webhook (e.g. an n8n workflow).

If TICKET_WEBHOOK_URL is empty the call is a dry run and returns a fake id.
"""

import json
import uuid
from typing import Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agent_starter.config import get_settings
from agent_starter.tools import register
from agent_starter.tools._http import request_with_retry


class TicketArgs(BaseModel):
    title: str = Field(max_length=120)
    description: str = Field(max_length=4000)
    priority: Literal["low", "medium", "high"] = "medium"


async def create_ticket(title: str, description: str, priority: str = "medium") -> str:
    s = get_settings()
    payload = {"title": title, "description": description, "priority": priority}
    if not s.ticket_webhook_url:
        return json.dumps({"ticket_id": f"DRY-{uuid.uuid4().hex[:6]}", "status": "dry_run"})
    resp = await request_with_retry(
        "POST",
        s.ticket_webhook_url,
        json=payload,
        timeout_s=s.http_timeout_s,
        retries=s.http_retries,
    )
    resp.raise_for_status()
    return json.dumps({"status": "created", "response": resp.text[:500]})


register(
    StructuredTool.from_function(
        coroutine=create_ticket,
        name="create_ticket",
        description="Create a support ticket when the user asks to open/raise/escalate a ticket.",
        args_schema=TicketArgs,
    )
)
