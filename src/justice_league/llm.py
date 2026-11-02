"""Cliente de modelos. Hoy: Ollama por HTTP. Mañana: cualquier servidor compatible.

Concepto: *Protocol* (tipado estructural). `ChatModel` describe la forma que debe
tener un modelo (un método `chat`) sin obligar a heredar de nada. Así el agente
funciona igual con Ollama, con llama.cpp, o con un modelo falso en los tests.
"""

from __future__ import annotations

import json
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

        message = response.json().get("message", {})
        return ModelReply(
            content=message.get("content") or "",
            tool_calls=[_parse_tool_call(c) for c in message.get("tool_calls") or []],
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
        try:
            with httpx.stream("POST", f"{self.host}/api/chat", json=payload, timeout=self.timeout) as response:
                if response.status_code >= 400:
                    response.read()
                    raise OllamaError(f"Ollama respondió {response.status_code}: {response.text[:300]}")
                for line in response.iter_lines():
                    if not line:
                        continue
                    message = json.loads(line).get("message", {})
                    if text := message.get("content"):
                        parts.append(text)
                        yield text
                    tool_calls.extend(_parse_tool_call(c) for c in message.get("tool_calls") or [])
        except httpx.ConnectError as exc:
            raise OllamaError(f"No pude conectar con Ollama en {self.host}. ¿Está corriendo (`ollama serve`)?") from exc
        yield ModelReply(content="".join(parts), tool_calls=tool_calls)


def _parse_tool_call(raw: dict[str, Any]) -> ToolCall:
    function = raw.get("function", {})
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):  # algunos servidores mandan los argumentos como texto JSON
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    return ToolCall(name=function.get("name", ""), arguments=arguments)
