import json

import httpx
import pytest
import respx
from langchain_core.tools import StructuredTool

from agent_starter.tools import get_tool, get_tools, register


def test_registry_discovers_tools() -> None:
    assert {t.name for t in get_tools()} >= {"http_get_json", "create_ticket"}


def test_duplicate_registration_rejected() -> None:
    async def noop() -> str:
        return ""

    dup = StructuredTool.from_function(coroutine=noop, name="create_ticket", description="x")
    with pytest.raises(ValueError, match="duplicate"):
        register(dup)


def test_unknown_tool() -> None:
    with pytest.raises(KeyError):
        get_tool("nope")


async def test_create_ticket_dry_run() -> None:
    out = json.loads(await get_tool("create_ticket").ainvoke({"title": "t", "description": "d"}))
    assert out["status"] == "dry_run"
    assert out["ticket_id"].startswith("DRY-")


@respx.mock
async def test_create_ticket_posts_to_webhook(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TICKET_WEBHOOK_URL", "https://hooks.test/ticket")
    route = respx.post("https://hooks.test/ticket").respond(200, json={"id": 7})
    args = {"title": "t", "description": "d", "priority": "high"}
    out = json.loads(await get_tool("create_ticket").ainvoke(args))
    assert out["status"] == "created"
    assert json.loads(route.calls.last.request.content)["priority"] == "high"


@respx.mock
async def test_http_tool_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTP_API_BASE_URL", "https://api.test")
    route = respx.get("https://api.test/status")
    route.side_effect = [httpx.Response(503), httpx.Response(200, json={"ok": True})]
    out = await get_tool("http_get_json").ainvoke({"path": "/status"})
    assert out == 'status=200 body={"ok": true}'
    assert route.call_count == 2


@respx.mock
async def test_http_tool_gives_up_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTP_API_BASE_URL", "https://api.test")
    monkeypatch.setenv("HTTP_RETRIES", "1")
    respx.get("https://api.test/down").mock(side_effect=httpx.ConnectTimeout("timeout"))
    with pytest.raises(httpx.ConnectTimeout):
        await get_tool("http_get_json").ainvoke({"path": "/down"})


async def test_http_tool_rejects_bad_path() -> None:
    out = await get_tool("http_get_json").ainvoke({"path": "https://evil.test"})
    assert out.startswith("error:")
