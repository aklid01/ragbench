"""A fake OpenAI-compatible chat completion server with scripted answers.

The server replays the scripted answers in order (the last one repeats when
the script runs out) and records every request body, so tests can assert on
the prompt that actually went over the wire.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _Handler(BaseHTTPRequestHandler):
    def __init__(self, *args: object, state: FakeLLMServer, **kwargs: object) -> None:
        self.state = state
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        self.state.received.append(body)
        content = self.state.next_reply()

        payload = {
            "id": "chatcmpl-fake",
            "model": body.get("model", "fake-llm"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 321, "completion_tokens": 54, "total_tokens": 375},
        }
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        pass


class FakeLLMServer:
    """Run the fake server on an ephemeral local port."""

    def __init__(self, replies: str | list[str]) -> None:
        self.received: list[dict] = []
        self._replies = [replies] if isinstance(replies, str) else list(replies)
        self._index = 0
        self._lock = threading.Lock()
        self._httpd = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            lambda *args, **kwargs: _Handler(*args, state=self, **kwargs),
        )
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def next_reply(self) -> str:
        with self._lock:
            content = self._replies[min(self._index, len(self._replies) - 1)]
            self._index += 1
        return content

    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}/v1"

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
