"""A fake OpenAI-compatible embeddings server with deterministic vectors.

The vector for a text is a bag of hashed word tokens: every word lands on a
fixed dimension with a fixed sign, so texts that share words end up close
under cosine distance. Good enough for retrieval to be meaningfully asserted,
cheap enough to run in every test.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_TOKEN = re.compile(r"[a-z0-9]+")


def hash_embedding(text: str, dimensions: int = 64) -> list[float]:
    vector = [0.0] * dimensions
    for token in _TOKEN.findall(text.lower()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = digest[0] % dimensions
        sign = 1.0 if digest[1] % 2 == 0 else -1.0
        vector[index] += sign
    norm = sum(component**2 for component in vector) ** 0.5
    if norm:
        vector = [component / norm for component in vector]
    return vector


class _Handler(BaseHTTPRequestHandler):
    def __init__(self, *args: object, state: FakeEmbeddingServer, **kwargs: object) -> None:
        self.state = state
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        self.state.request_count += 1
        self.state.received_texts.extend(body["input"])

        payload = {
            "object": "list",
            "model": body.get("model", "fake"),
            "data": [
                {"object": "embedding", "index": index, "embedding": hash_embedding(text)}
                for index, text in enumerate(body["input"])
            ],
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        pass


class FakeEmbeddingServer:
    """Run the fake server on an ephemeral local port."""

    def __init__(self) -> None:
        self.request_count = 0
        self.received_texts: list[str] = []
        self._httpd = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            lambda *args, **kwargs: _Handler(*args, state=self, **kwargs),
        )
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}/v1"

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
