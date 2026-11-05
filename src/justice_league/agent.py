"""El bucle del agente. Es el corazón del proyecto y cabe en una pantalla.

    pregunta → modelo → ¿pidió herramientas?
                          ├─ no → respuesta final
                          └─ sí → (pedir permiso) → ejecutar → devolver resultado → modelo ...

Concepto: *generador como flujo de eventos*. `run()` no imprime nada ni sabe de
interfaces: va "emitiendo" eventos (llamada a herramienta, resultado, respuesta)
y quien lo consuma decide cómo mostrarlos. Los eventos "token" son fragmentos de
texto en vivo; "stats" trae las métricas de cada llamada al modelo; "answer" sigue llegando al final con el texto completo; "summary" cierra el turno con sus
tiempos y el contexto usado. La CLI los pinta con Rich; Atalaya,
como pasos del chat. Ese es el límite entre el núcleo y los adaptadores.
"""

from __future__ import annotations

import json
import time
from collections.abc import Generator, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

from .llm import ChatModel, ModelReply, ToolCall
from .permissions import Approver
from .tools import ToolRegistry


# Cuántas veces se tolera que el modelo insista con una llamada idéntica que ya falló.
MAX_REPEATED_CALLS = 2


@dataclass
class Event:
    kind: Literal["token", "stats", "notice", "tool_call", "tool_result", "answer", "error", "summary"]
    text: str = ""
    name: str = ""
    args: dict[str, Any] = field(default_factory=dict)


class Agent:
    def __init__(
        self,
        model: ChatModel,
        tools: ToolRegistry,
        approve: Approver,
        system_prompt: str,
        max_steps: int = 10,
    ):
        self.model = model
        self.tools = tools
        self.approve = approve
        self.max_steps = max_steps
        self._notice = ""
        self._replies: list[ModelReply] = []  # llamadas al modelo del turno en curso
        self._waited = 0.0  # segundos esperando al usuario (aprobaciones) en el turno en curso
        # El historial vive en el agente: una instancia = una conversación.
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    def reset(self) -> None:
        self.messages = self.messages[:1]
        self._notice = ""
        if self.tools.staging:
            self.tools.staging.discard()

    def run(self, user_text: str) -> Iterator[Event]:
        staging = self.tools.staging
        if staging:
            staging.discard()  # restos de un turno interrumpido: nunca se aprobaron
        # Aviso sobre el desenlace del turno anterior (cambios aplicados o rechazados), para que
        # el modelo no crea otra cosa. Va dentro del mensaje del usuario para no romper la alternancia.
        text = f"{self._notice}\n\n{user_text}" if self._notice else user_text
        self._notice = ""
        self.messages.append({"role": "user", "content": text})

        started = time.perf_counter()
        self._replies, self._waited = [], 0.0
        yield from self._turn()
        yield from self._flush()
        if any(r.eval_tokens for r in self._replies):  # sin métricas (p. ej. modelo falso) no hay resumen
            yield Event("summary", args=self._summary(time.perf_counter() - started - self._waited))

    def _summary(self, seconds: float) -> dict[str, Any]:
        """Cifras del turno completo. El tiempo no cuenta lo que el usuario tardó en aprobar."""
        first, last = self._replies[0], self._replies[-1]
        return {
            "seconds": seconds,
            "calls": len(self._replies),
            "load_seconds": sum(r.load_seconds for r in self._replies),
            "first_token_seconds": first.first_token_seconds,
            "eval_tokens": sum(r.eval_tokens for r in self._replies),
            "eval_seconds": sum(r.eval_seconds for r in self._replies),
            # Lo que ocupa la conversación tras la última llamada: su prompt + lo que generó.
            "context_tokens": last.prompt_tokens + last.eval_tokens,
            "num_ctx": getattr(self.model, "num_ctx", 0),
        }

    def _approve(self, name: str, args: dict[str, Any], preview: str | None) -> bool:
        waiting = time.perf_counter()
        try:
            return self.approve(name, args, preview)
        finally:
            self._waited += time.perf_counter() - waiting

    def _flush(self) -> Iterator[Event]:
        """Fin del turno: si el modelo dejó cambios preparados, pide UNA aprobación con el diff combinado."""
        staging = self.tools.staging
        if staging is None or not staging.pending():
            return
        names = ", ".join(staging.names())
        if self._approve("aplicar_cambios", {"archivos": staging.names()}, staging.diff()):
            staging.apply()
            yield Event("notice", f"Cambios aplicados: {names}")
            self._notice = f"[Sistema: el usuario aprobó y se aplicaron los cambios en {names}.]"
        else:
            staging.discard()
            yield Event("notice", "Cambios descartados: no se escribió nada.")
            self._notice = f"[Sistema: el usuario rechazó los cambios en {names}; no se escribió nada. No insistas: pregunta qué prefiere.]"

    def _turn(self) -> Iterator[Event]:
        # Concepto: *cortafuegos de repetición*. Un modelo chico a veces repite la misma
        # llamada fallida una y otra vez. Aquí recordamos qué llamadas fallaron en este turno.
        failed: dict[str, str] = {}
        blocked = 0

        # `max_steps` evita bucles infinitos: un modelo chico puede quedarse llamando
        # la misma herramienta una y otra vez.
        for _ in range(self.max_steps):
            reply = yield from self._ask_model()
            self._replies.append(reply)
            if reply.eval_tokens:
                yield Event("stats", args=_metrics_of(reply))

            assistant_message: dict[str, Any] = {"role": "assistant", "content": reply.content}
            if reply.tool_calls:
                assistant_message["tool_calls"] = [
                    {"function": {"name": c.name, "arguments": c.arguments}} for c in reply.tool_calls
                ]
            self.messages.append(assistant_message)

            if not reply.tool_calls:
                yield Event("answer", reply.content)
                return

            for call in reply.tool_calls:
                yield Event("tool_call", name=call.name, args=call.arguments)
                key = f"{call.name}:{json.dumps(call.arguments, sort_keys=True, default=str)}"
                if key in failed:
                    blocked += 1
                    result = (
                        f"REPETIDO: ya llamaste a '{call.name}' con estos mismos argumentos y falló ({failed[key][:150]}). "
                        "No lo repitas: prueba otro enfoque o explícale el problema al usuario."
                    )
                else:
                    result = self._dispatch(call)
                    if result.startswith(("ERROR", "DENEGADO")):
                        failed[key] = result
                self.messages.append({"role": "tool", "tool_name": call.name, "content": result})
                yield Event("tool_result", result, call.name, call.arguments)
                if blocked >= MAX_REPEATED_CALLS:
                    yield Event("error", "Detuve el bucle: el modelo insistió con la misma llamada fallida.")
                    return

        yield Event("error", f"Me detuve tras {self.max_steps} pasos sin llegar a una respuesta.")

    def _ask_model(self) -> Generator[Event, None, ModelReply]:
        """Pide la respuesta al modelo; si sabe hacer streaming, reenvía cada fragmento como evento "token"."""
        specs = self.tools.specs()
        stream = getattr(self.model, "stream", None)
        if stream is None:
            return self.model.chat(self.messages, specs)
        for item in stream(self.messages, specs):
            if isinstance(item, str):
                yield Event("token", item)
            else:
                return item
        raise RuntimeError("El stream del modelo terminó sin respuesta final.")

    def _dispatch(self, call: ToolCall) -> str:
        """Ejecuta una llamada. Los errores vuelven al modelo como texto para que pueda corregirse."""
        tool = self.tools.get(call.name)
        if tool is None:
            return f"ERROR: la herramienta '{call.name}' no existe. Disponibles: {', '.join(self.tools.names())}"

        try:
            if tool.needs_approval:
                preview = tool.preview(**call.arguments) if tool.preview else None
                if not self._approve(call.name, call.arguments, preview):
                    return "DENEGADO: el usuario no aprobó esta acción. No la reintentes sin preguntar."
            return tool.fn(**call.arguments)
        except TypeError as exc:  # argumentos faltantes o con nombre equivocado
            return f"ERROR: argumentos inválidos para '{call.name}': {exc}"
        except Exception as exc:  # incluye SandboxError y errores de E/S
            return f"ERROR: {type(exc).__name__}: {exc}"


def _metrics_of(reply: ModelReply) -> dict[str, Any]:
    """Las métricas de una llamada (sin el texto ni las herramientas)."""
    return {
        "prompt_tokens": reply.prompt_tokens,
        "eval_tokens": reply.eval_tokens,
        "eval_seconds": reply.eval_seconds,
        "load_seconds": reply.load_seconds,
        "prompt_seconds": reply.prompt_seconds,
        "total_seconds": reply.total_seconds,
        "first_token_seconds": reply.first_token_seconds,
    }
