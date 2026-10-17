import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from app.parsing.llm_parser import parse_sections
from app.parsing.ollama_client import OllamaClient


class Handler(BaseHTTPRequestHandler):
    seen = {}

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.seen = {"path": self.path, "body": body, "auth": self.headers.get("Authorization")}
        out = json.dumps({"message": {"role": "assistant", "content": '{"skills": ["Python", "SQL"]}'}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_request_shape_and_response(server):
    client = OllamaClient("qwen2.5:7b", host=server)
    resp = client.chat_completion(messages=[{"role": "user", "content": "hi"}], max_tokens=50, temperature=0)

    assert resp.choices[0].message.content == '{"skills": ["Python", "SQL"]}'
    seen = Handler.seen
    assert seen["path"] == "/api/chat"
    assert seen["body"]["model"] == "qwen2.5:7b"
    assert seen["body"]["stream"] is False
    assert seen["body"]["format"] == "json"
    assert seen["body"]["options"]["num_predict"] == 50


def test_parse_sections_works_with_ollama_client(server):
    body = parse_sections({"skills": "Python, SQL"}, OllamaClient("qwen2.5:7b", host=server))
    assert body.skills == ["Python", "SQL"]


def test_clear_error_when_ollama_not_running():
    client = OllamaClient("qwen2.5:7b", host="http://127.0.0.1:1")
    with pytest.raises(RuntimeError, match="Ollama"):
        client.chat_completion(messages=[{"role": "user", "content": "hi"}])


def test_cloud_mode_sends_api_key_and_skips_num_ctx(server):
    client = OllamaClient("gpt-oss:120b", host=server, api_key="sk-test")
    client.chat_completion(messages=[{"role": "user", "content": "hi"}])
    assert Handler.seen["auth"] == "Bearer sk-test"
    assert "num_ctx" not in Handler.seen["body"]["options"]


def test_local_mode_has_no_auth_header_and_sets_num_ctx(server):
    client = OllamaClient("qwen2.5:7b", host=server)
    client.chat_completion(messages=[{"role": "user", "content": "hi"}])
    assert Handler.seen["auth"] is None
    assert Handler.seen["body"]["options"]["num_ctx"] == 8192


def test_get_client_reads_cloud_settings(monkeypatch):
    from app.parsing.llm_parser import get_client

    monkeypatch.setenv("LLM_BACKEND", "ollama")
    monkeypatch.setenv("OLLAMA_HOST", "https://ollama.com")
    monkeypatch.setenv("OLLAMA_API_KEY", "sk-abc")
    monkeypatch.setenv("OLLAMA_MODEL", "some-cloud-model")
    c = get_client()
    assert (c.host, c.api_key, c.model) == ("https://ollama.com", "sk-abc", "some-cloud-model")