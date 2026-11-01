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
