"""Pruebas del cliente de Ollama con respuestas HTTP simuladas (sin servidor real)."""

import httpx
import pytest

from justice_league.llm import OllamaChat, OllamaError


def respuesta(json_body, status=200):
    return httpx.Response(status, json=json_body, request=httpx.Request("POST", "http://x/api/chat"))


def test_parsea_texto_y_llamadas_a_herramientas(monkeypatch):
    cuerpo = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "a.txt"}}}],
        }
    }
    capturado = {}

    def falso_post(url, json, timeout):
        capturado.update(url=url, payload=json)
        return respuesta(cuerpo)

    monkeypatch.setattr(httpx, "post", falso_post)
    reply = OllamaChat("qwen2.5:14b", num_ctx=4096).chat([{"role": "user", "content": "x"}], tools=[{"type": "function"}])

    assert reply.tool_calls[0].name == "read_file" and reply.tool_calls[0].arguments == {"path": "a.txt"}
    assert capturado["url"].endswith("/api/chat")
    assert capturado["payload"]["options"]["num_ctx"] == 4096
    assert capturado["payload"]["stream"] is False


def test_argumentos_como_texto_json(monkeypatch):
    cuerpo = {"message": {"content": "", "tool_calls": [{"function": {"name": "t", "arguments": '{"a": 1}'}}]}}
    monkeypatch.setattr(httpx, "post", lambda *a, **k: respuesta(cuerpo))
    assert OllamaChat("m").chat([]).tool_calls[0].arguments == {"a": 1}


def test_ollama_apagado_da_error_claro(monkeypatch):
    def falla(*a, **k):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(httpx, "post", falla)
    with pytest.raises(OllamaError, match="ollama serve"):
        OllamaChat("m").chat([])


def test_error_http(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: respuesta({"error": "model not found"}, status=404))
    with pytest.raises(OllamaError, match="404"):
        OllamaChat("m").chat([])


def test_stream_emite_fragmentos_y_respuesta_final(monkeypatch):
    lineas = [
        '{"message": {"content": "Ho"}}',
        '{"message": {"content": "la"}}',
        '{"message": {"content": "", "tool_calls": [{"function": {"name": "t", "arguments": {"a": 1}}}]}, "done": true}',
    ]
    capturado = {}

    class FalsoStream:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def iter_lines(self):
            return iter(lineas)

    def falso_stream(method, url, json, timeout):
        capturado["payload"] = json
        return FalsoStream()

    monkeypatch.setattr(httpx, "stream", falso_stream)
    items = list(OllamaChat("m").stream([{"role": "user", "content": "x"}]))

    assert items[:2] == ["Ho", "la"]
    assert items[2].content == "Hola" and items[2].tool_calls[0].arguments == {"a": 1}
    assert capturado["payload"]["stream"] is True
