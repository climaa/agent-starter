import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from agent_starter.api.main import app
from agent_starter.config import get_settings
from agent_starter.graph.build import Graph


@pytest.fixture
def client(graph: Graph) -> Iterator[TestClient]:
    app.state.graph = graph
    with TestClient(app) as c:
        yield c
    del app.state.graph


def parse_sse(body: str) -> list[tuple[str, dict[str, object]]]:
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in fields:
            events.append((fields["event"], json.loads(fields["data"])))
    return events


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.json() == {"status": "ok", "agent_ready": True}
    assert r.headers["x-request-id"]


def test_chat_streams_nodes_tokens_and_final(client: TestClient) -> None:
    r = client.post("/chat", json={"message": "How many days to request a refund?"})
    assert r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    nodes = [d["node"] for e, d in events if e == "node"]
    assert nodes == ["guard_input", "plan", "retrieve", "act", "answer", "guard_output"]
    assert any(e == "token" for e, _ in events)
    final = events[-1]
    assert final[0] == "final" and "30 days" in str(final[1]["answer"])


def test_n8n_webhook(client: TestClient) -> None:
    r = client.post("/webhook/n8n", json={"message": "refund days?", "thread_id": "n8n-1"})
    body = r.json()
    assert body["thread_id"] == "n8n-1"
    assert body["sources"][0]["source"] == "refunds.md"


def test_n8n_webhook_token(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("N8N_WEBHOOK_TOKEN", "s3cret")
    get_settings.cache_clear()
    assert client.post("/webhook/n8n", json={"message": "hi"}).status_code == 401
    ok = client.post("/webhook/n8n", json={"message": "hi"}, headers={"X-Webhook-Token": "s3cret"})
    assert ok.status_code == 200
