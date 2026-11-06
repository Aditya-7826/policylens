"""API contract: SSE event order and validation. The pipeline is stubbed (no models)."""

import json

from fastapi.testclient import TestClient

import api.main as main


def test_ask_streams_stages_then_result(monkeypatch):
    def fake_stream(question, config):
        yield "stage", {"stage": "analyze", "label": "Analysed the question"}
        yield "stage", {"stage": "retrieve", "label": "Retrieved", "evidence": 3}
        yield "result", {"question": question, "answer": "ok [GDPR, Article 33, gdpr::article-33::0]"}

    monkeypatch.setattr(main, "stream_ask", fake_stream)
    client = TestClient(main.app)
    with client.stream("POST", "/ask", json={"question": "What is Article 33?"}) as r:
        assert r.status_code == 200
        body = "".join(r.iter_text())
    events = [line.split(": ", 1)[1] for line in body.splitlines() if line.startswith("event: ")]
    assert events == ["stage", "stage", "result"]
    data = [json.loads(line.split(": ", 1)[1]) for line in body.splitlines() if line.startswith("data: ")]
    assert data[-1]["answer"].startswith("ok")


def test_ask_errors_become_events(monkeypatch):
    def boom(question, config):
        raise RuntimeError("all LLM providers failed")
        yield  # pragma: no cover

    monkeypatch.setattr(main, "stream_ask", boom)
    with TestClient(main.app).stream("POST", "/ask", json={"question": "anything here"}) as r:
        body = "".join(r.iter_text())
    assert "event: error" in body and "all LLM providers failed" in body


def test_validation():
    client = TestClient(main.app)
    assert client.post("/ask", json={"question": "x"}).status_code == 422
    assert client.post("/ask", json={"question": "valid question", "config": "nope"}).status_code == 400
    r = client.post("/ingest", files=[("files", ("evil.exe", b"MZ"))])
    assert r.status_code == 400
