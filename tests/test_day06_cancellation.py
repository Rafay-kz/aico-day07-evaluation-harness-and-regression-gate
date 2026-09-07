"""Client cancellation must abort in-flight fake model/RAG work."""

from __future__ import annotations

import asyncio
import json
import threading

import pytest

from aico.api.app import create_app
from aico.api.dependencies import get_trusted_claims
from aico.platform.errors import OperationCancelled
from aico.platform.fake_transport import FakeTransport
from tests.day06_helpers import VALID_CLAIMS, make_container, supported_chat_json


class SlowFakeTransport(FakeTransport):
    def __init__(self) -> None:
        super().__init__(wait_for_cancel=True, chat_texts=[supported_chat_json()])
        self.entered_chat = threading.Event()
        self.saw_cancel = threading.Event()

    def chat(self, messages, *, model_alias, timeout_seconds, cancellation):
        self.entered_chat.set()
        try:
            return super().chat(
                messages,
                model_alias=model_alias,
                timeout_seconds=timeout_seconds,
                cancellation=cancellation,
            )
        except OperationCancelled:
            self.saw_cancel.set()
            raise


@pytest.mark.asyncio
async def test_http_disconnect_cancels_in_flight_gateway_work() -> None:
    transport = SlowFakeTransport()
    container, service, _ = make_container(transport=transport)
    application = create_app(container)
    application.dependency_overrides[get_trusted_claims] = lambda: VALID_CLAIMS

    body = json.dumps({"question": "Synthetic cancellation question"}).encode()
    headers = [
        (b"host", b"test"),
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
    ]
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/ask",
        "raw_path": b"/ask",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 50000),
        "server": ("test", 80),
    }
    messages: asyncio.Queue = asyncio.Queue()
    messages.put_nowait({"type": "http.request", "body": body, "more_body": False})
    sent: list[dict] = []

    async def receive() -> dict:
        return await messages.get()

    async def send(message: dict) -> None:
        sent.append(message)

    task = asyncio.create_task(application(scope, receive, send))
    assert await asyncio.to_thread(transport.entered_chat.wait, 3)
    await messages.put({"type": "http.disconnect"})
    await asyncio.wait_for(task, timeout=3)
    assert await asyncio.to_thread(transport.saw_cancel.wait, 3)
    statuses = [item.get("status") for item in sent if item.get("type") == "http.response.start"]
    assert statuses
    assert statuses[0] != 200
    bodies = b"".join(
        item.get("body", b"") for item in sent if item.get("type") == "http.response.body"
    )
    assert b"answered" not in bodies
    assert transport.chat_calls == 0
    assert service.calls == ["Synthetic cancellation question"]
