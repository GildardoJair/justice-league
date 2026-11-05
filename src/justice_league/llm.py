"""Cliente de modelos. Hoy: Ollama por HTTP. Mañana: cualquier servidor compatible.

Concepto: *Protocol* (tipado estructural). `ChatModel` describe la forma que debe
tener un modelo (un método `chat`) sin obligar a heredar de nada. Así el agente
funciona igual con Ollama, con llama.cpp, o con un modelo falso en los tests.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelReply:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Métricas que Ollama reporta al terminar (0 si no las dio, p. ej. en los tests).
    prompt_tokens: int = 0
    eval_tokens: int = 0
    eval_seconds: float = 0.0
    load_seconds: float = 0.0  # cargar el modelo en memoria (~0,1 s si ya estaba cargado)
    prompt_seconds: float = 0.0  # leer el prompt (prefill); baja mucho si Ollama lo tenía en cache
    total_seconds: float = 0.0  # todo lo que tardó Ollama en esta llamada
    # Medido por nosotros: desde que se envía la petición hasta el primer fragmento (solo en streaming).
    first_token_seconds: float | None = None


class ChatModel(Protocol):
    """Lo mínimo es `chat`. Si además tiene `stream`, el agente lo prefiere (ver Agent.run)."""

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelReply: ...


class OllamaError(RuntimeError):
    pass


class OllamaChat:
    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        num_ctx: int = 8192,
        temperature: float = 0.2,
        timeout: float = 300.0,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.timeout = timeout

    def _payload(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            # num_ctx fija cuántos tokens de contexto reserva Ollama. Más contexto = más
            # VRAM para el KV cache. Es la perilla que vamos a medir en EXPERIMENTS.md.
            "options": {"num_ctx": self.num_ctx, "temperature": self.temperature},
        }
        if tools:
            payload["tools"] = tools
        return payload

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelReply:
        payload = self._payload(messages, tools, stream=False)
        try:
            response = httpx.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise OllamaError(f"No pude conectar con Ollama en {self.host}. ¿Está corriendo (`ollama serve`)?") from exc
        except httpx.HTTPStatusError as exc:
            raise OllamaError(f"Ollama respondió {exc.response.status_code}: {exc.response.text[:300]}") from exc

        body = response.json()
        message = body.get("message", {})
        return ModelReply(
            content=message.get("content") or "",
            tool_calls=[_parse_tool_call(c) for c in message.get("tool_calls") or []],
            **_metrics(body),
        )

    def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> Iterator[str | ModelReply]:
        """Como `chat`, pero emite cada fragmento de texto (`str`) al llegar y, al final, el `ModelReply` completo.

        Concepto: *streaming NDJSON*. Con `stream: true` Ollama manda una línea JSON por
        fragmento en vez de una sola respuesta al terminar. Las llamadas a herramientas
        llegan enteras en algún fragmento, así que solo el texto se emite en vivo.
        """
        payload = self._payload(messages, tools, stream=True)
        parts: list[str] = []
        tool_calls: list[ToolCall] = []
        metrics: dict[str, Any] = {}
        started = time.perf_counter()
        first_token: float | None = None
        try:
            with httpx.stream("POST", f"{self.host}/api/chat", json=payload, timeout=self.timeout) as response:
                if response.status_code >= 400:
                    response.read()
                    raise OllamaError(f"Ollama respondió {response.status_code}: {response.text[:300]}")
                for line in response.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    message = chunk.get("message", {})
                    if first_token is None and (message.get("content") or message.get("tool_calls")):
                        first_token = time.perf_counter() - started
                    if chunk.get("done"):
                        metrics = _metrics(chunk)
                    if text := message.get("content"):
                        parts.append(text)
                        yield text
                    tool_calls.extend(_parse_tool_call(c) for c in message.get("tool_calls") or [])
        except httpx.ConnectError as exc:
            raise OllamaError(f"No pude conectar con Ollama en {self.host}. ¿Está corriendo (`ollama serve`)?") from exc
        yield ModelReply(content="".join(parts), tool_calls=tool_calls, first_token_seconds=first_token, **metrics)


def list_models(host: str = "http://localhost:11434") -> list[str]:
    """Nombres de los modelos descargados en Ollama (vacío si no responde)."""
    try:
        response = httpx.get(f"{host.rstrip('/')}/api/tags", timeout=5)
        response.raise_for_status()
    except httpx.HTTPError:
        return []
    return sorted(m["name"] for m in response.json().get("models", []))


# Cliente HTTP compartido para las consultas frecuentes (/api/ps): crear uno nuevo en cada
# llamada (lo que hace `httpx.get`) cuesta ~17 ms de CPU por su contexto SSL; reutilizarlo, <1 ms.
_poll_client = httpx.Client(timeout=2)


def running_models(host: str = "http://localhost:11434") -> list[dict[str, Any]] | None:
    """Lo que Ollama tiene cargado en memoria ahora mismo (`/api/ps`), con su detalle; None si no responde.

    Un modelo aparece aquí solo cuando ya terminó de cargar (comprobado con Ollama 0.21.2).
    """
    try:
        response = _poll_client.get(f"{host.rstrip('/')}/api/ps")
        response.raise_for_status()
    except httpx.HTTPError:
        return None
    return response.json().get("models", [])


def loaded_models(host: str = "http://localhost:11434") -> list[str] | None:
    """Solo los nombres de `running_models`."""
    models = running_models(host)
    return None if models is None else [m["name"] for m in models]


def model_loaded(model: str, host: str = "http://localhost:11434") -> bool | None:
    """¿Está `model` cargado en memoria? None si Ollama no responde. `llama3.2` equivale a `llama3.2:latest`."""
    loaded = loaded_models(host)
    if loaded is None:
        return None
    full = model if ":" in model else f"{model}:latest"
    return full in loaded


def _metrics(body: dict[str, Any]) -> dict[str, Any]:
    """Cifras del último mensaje de Ollama. Las duraciones vienen en nanosegundos.

    `prompt_eval_count` cuenta el prompt completo aunque Ollama reutilice su cache
    (comprobado con Ollama 0.21.2), así que sirve para medir cuánto contexto se usa.
    """
    return {
        "prompt_tokens": body.get("prompt_eval_count") or 0,
        "eval_tokens": body.get("eval_count") or 0,
        "eval_seconds": (body.get("eval_duration") or 0) / 1e9,
        "load_seconds": (body.get("load_duration") or 0) / 1e9,
        "prompt_seconds": (body.get("prompt_eval_duration") or 0) / 1e9,
        "total_seconds": (body.get("total_duration") or 0) / 1e9,
    }


def _parse_tool_call(raw: dict[str, Any]) -> ToolCall:
    function = raw.get("function", {})
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):  # algunos servidores mandan los argumentos como texto JSON
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    return ToolCall(name=function.get("name", ""), arguments=arguments)
