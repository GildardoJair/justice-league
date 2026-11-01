"""Cliente de modelos. Hoy: Ollama por HTTP. Mañana: cualquier servidor compatible.

Concepto: *Protocol* (tipado estructural). `ChatModel` describe la forma que debe
tener un modelo (un método `chat`) sin obligar a heredar de nada. Así el agente
funciona igual con Ollama, con llama.cpp, o con un modelo falso en los tests.
"""

from __future__ import annotations

import json
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

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelReply:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            # num_ctx fija cuántos tokens de contexto reserva Ollama. Más contexto = más
            # VRAM para el KV cache. Es la perilla que vamos a medir en EXPERIMENTS.md.
            "options": {"num_ctx": self.num_ctx, "temperature": self.temperature},
        }
        if tools:
            payload["tools"] = tools
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


def _parse_tool_call(raw: dict[str, Any]) -> ToolCall:
    function = raw.get("function", {})
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):  # algunos servidores mandan los argumentos como texto JSON
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    return ToolCall(name=function.get("name", ""), arguments=arguments)
