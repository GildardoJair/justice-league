"""Pruebas del cliente de Ollama con respuestas HTTP simuladas (sin servidor real)."""

import httpx
import pytest

from justice_league import llm
from justice_league.llm import OllamaChat, OllamaError, list_models, loaded_models, model_loaded


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
    assert items[2].first_token_seconds is not None and items[2].first_token_seconds >= 0


def test_chat_lee_metricas(monkeypatch):
    cuerpo = {"message": {"content": "x"}, "prompt_eval_count": 50, "eval_count": 20, "eval_duration": 2_000_000_000}
    monkeypatch.setattr(httpx, "post", lambda *a, **k: respuesta(cuerpo))
    reply = OllamaChat("m").chat([])
    assert (reply.prompt_tokens, reply.eval_tokens, reply.eval_seconds) == (50, 20, 2.0)


def test_chat_lee_tiempos_de_carga_prompt_y_total(monkeypatch):
    cuerpo = {"message": {"content": "x"}, "load_duration": 5_700_000_000,
              "prompt_eval_duration": 400_000_000, "total_duration": 9_000_000_000}
    monkeypatch.setattr(httpx, "post", lambda *a, **k: respuesta(cuerpo))
    reply = OllamaChat("m").chat([])
    assert (reply.load_seconds, reply.prompt_seconds, reply.total_seconds) == (5.7, 0.4, 9.0)
    assert reply.first_token_seconds is None  # sin streaming no se mide


def test_list_models(monkeypatch):
    cuerpo = {"models": [{"name": "qwen2.5:14b"}, {"name": "llama3.2:latest"}]}
    monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(200, json=cuerpo, request=httpx.Request("GET", "http://x")))
    assert list_models() == ["llama3.2:latest", "qwen2.5:14b"]


def test_list_models_sin_ollama(monkeypatch):
    def falla(*a, **k):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(httpx, "get", falla)
    assert list_models() == []


def test_loaded_models(monkeypatch):
    cuerpo = {"models": [{"name": "qwen2.5:14b"}]}
    monkeypatch.setattr(llm._poll_client, "get", lambda *a, **k: httpx.Response(200, json=cuerpo, request=httpx.Request("GET", "http://x/api/ps")))
    assert loaded_models() == ["qwen2.5:14b"]


def test_loaded_models_sin_ollama(monkeypatch):
    def caido(*a, **k):
        raise httpx.ConnectError("no")

    monkeypatch.setattr(llm._poll_client, "get", caido)
    assert loaded_models() is None


def test_model_loaded_entiende_latest(monkeypatch):
    cuerpo = {"models": [{"name": "llama3.2:latest"}]}
    monkeypatch.setattr(llm._poll_client, "get", lambda *a, **k: httpx.Response(200, json=cuerpo, request=httpx.Request("GET", "http://x/api/ps")))
    assert model_loaded("llama3.2") is True and model_loaded("qwen2.5:14b") is False
