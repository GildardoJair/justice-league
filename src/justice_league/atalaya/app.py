"""Atalaya: chat web con Chainlit.

Se lanza con `jl atalaya` (o `chainlit run src/justice_league/atalaya/app.py`).
Carpeta de trabajo: variable de entorno JL_WORKSPACE (por defecto, la actual).
Aspecto (tema, logo, avatares, bienvenida): carpetas `.chainlit/` y `public/` junto a este archivo;
`jl atalaya` apunta Chainlit hacia aquí con CHAINLIT_APP_ROOT.

Concepto: *puente sync ↔ async*. El agente es síncrono (httpx bloqueante), pero
Chainlit es asíncrono. Corremos cada paso del agente en un hilo (`make_async`) para
no congelar la interfaz, y cuando el guardián necesita preguntar, el hilo le pide al
bucle de eventos que muestre los botones y espera la respuesta.

Nota: verificado solo por lectura contra la API de Chainlit 2.x; falta probarlo en tu
máquina con Ollama. Si la API cambió, el único punto frágil es `_ask_approval`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any

import chainlit as cl
from chainlit.input_widget import Select
from chainlit.server import app as server

from justice_league.config import load_settings
from justice_league.cyborg import snapshot
from justice_league.factory import build_agent
from justice_league.llm import OllamaError, list_models, model_loaded
from justice_league.ui import context_level, summary_parts

ROLE = "superman"
CYBORG_URL = "/jl/cyborg"  # el panel de Cyborg consulta aquí el estado del equipo


@server.get(CYBORG_URL, include_in_schema=False)
async def cyborg_snapshot() -> dict[str, Any]:
    # En un hilo: leer sensores y preguntar a Ollama son llamadas bloqueantes (~2 ms).
    return await asyncio.to_thread(snapshot, load_settings().ollama_host)


def _before_catch_all(path: str) -> None:
    """Chainlit responde cualquier ruta desconocida con su página (`/{full_path:path}`), y esa
    ruta se registró antes que la nuestra. FastAPI prueba las rutas en orden: movemos la nuestra
    al principio (si el módulo se recarga, se queda solo la última versión)."""
    routes = server.router.routes
    ours = [r for r in routes if getattr(r, "path", None) == path]
    for route in ours:
        routes.remove(route)
    routes.insert(0, ours[-1])


_before_catch_all(CYBORG_URL)


def cyborg_element() -> cl.CustomElement:
    # display="side": va en el panel lateral derecho. Chainlit cierra ese panel cada vez que
    # cambian los elementos de la página (los chips, el "Pensando…") salvo que haya alguno
    # "side" colgado de un mensaje; por eso Cyborg va adjunto al mensaje de bienvenida.
    return cl.CustomElement(name="Cyborg", display="side", props={"url": CYBORG_URL, "interval_ms": 2000})


async def open_cyborg() -> None:
    """Reabre el panel de Cyborg (lo usa el comando /cyborg)."""
    await cl.ElementSidebar.set_title("Cyborg")
    await cl.ElementSidebar.set_elements([cyborg_element()], key="cyborg")


class Waiting:
    """Indicador en vivo mientras el modelo aún no dice nada: "Cargando <modelo>…" o "Pensando…".

    Concepto: *tarea en segundo plano*. El agente espera a Ollama en su hilo; mientras
    tanto, una tarea del bucle de eventos consulta `/api/ps` para saber cuándo terminó la
    carga y cambiar el texto. El contador de segundos lo lleva el navegador (TurnStatus.jsx).
    """

    POLL_SECONDS = 0.5

    def __init__(self, host: str):
        self.host = host
        self._message: cl.Message | None = None
        self._task: asyncio.Task[None] | None = None

    async def start(self, model: str) -> None:
        await self.stop()
        started = time.perf_counter()
        loading = await cl.make_async(model_loaded)(model, self.host) is False
        element = cl.CustomElement(name="TurnStatus", display="inline", props=self._props(loading, model, started))
        self._message = cl.Message(content="", elements=[element])
        await self._message.send()
        if loading:
            self._task = asyncio.create_task(self._watch_load(element, model, started))

    async def _watch_load(self, element: cl.CustomElement, model: str, started: float) -> None:
        while await cl.make_async(model_loaded)(model, self.host) is False:
            await asyncio.sleep(self.POLL_SECONDS)
        element.props = self._props(False, model, started)
        await element.update()

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._message is not None:
            await self._message.remove()
            self._message = None

    @staticmethod
    def _props(loading: bool, model: str, started: float) -> dict[str, Any]:
        return {"phase": "loading" if loading else "thinking", "model": model, "elapsed": time.perf_counter() - started}


async def _ask_approval(name: str, args: dict[str, Any], preview: str | None) -> bool:
    body = f"```diff\n{preview}\n```" if preview else f"```json\n{json.dumps(args, ensure_ascii=False, indent=2)}\n```"
    answer = await cl.AskActionMessage(
        content=f"🛡️ **Permiso solicitado: `{name}`**\n\n{body}",
        actions=[
            cl.Action(name="approve", payload={"ok": True}, label="✅ Aprobar"),
            cl.Action(name="deny", payload={"ok": False}, label="❌ Denegar"),
        ],
        timeout=300,
    ).send()
    return bool(answer and answer.get("payload", {}).get("ok"))


@cl.set_starters
async def set_starters() -> list[cl.Starter]:
    return [
        cl.Starter(label="Explorar la carpeta", message="Lista los archivos de la carpeta de trabajo y dime qué contiene."),
        cl.Starter(label="Resumir un archivo", message="Lee el README.md y hazme un resumen breve."),
        cl.Starter(label="Revisar mi código", message="Revisa los archivos .py de la carpeta y dime si ves errores."),
        cl.Starter(label="Crear un archivo", message="Crea un archivo notas.txt con una lista de tareas de ejemplo."),
    ]


@cl.on_chat_start
async def on_chat_start() -> None:
    settings = load_settings()
    workspace = Path(os.environ.get("JL_WORKSPACE", ".")).resolve()
    loop = asyncio.get_running_loop()

    def approve(name: str, args: dict[str, Any], preview: str | None) -> bool:
        # Se llama desde el hilo del agente: delegamos la pregunta al bucle de Chainlit.
        return asyncio.run_coroutine_threadsafe(_ask_approval(name, args, preview), loop).result()

    agent = build_agent(settings, ROLE, workspace, approve)
    cl.user_session.set("agent", agent)
    cl.user_session.set("waiting", Waiting(settings.ollama_host))
    # Al llegar este mensaje, Chainlit abre solo el panel de Cyborg, y la palabra "Cyborg"
    # del texto queda como enlace para reabrirlo.
    await cl.Message(
        content=f"**{settings.roles[ROLE].name}** listo en `{workspace}`. Estado del equipo: Cyborg (o escribe /cyborg).",
        elements=[cyborg_element()],
    ).send()

    # Selector de modelo en el panel de ajustes: lista lo que Ollama tiene descargado.
    models = await cl.make_async(list_models)(settings.ollama_host)
    current = agent.model.model
    if current not in models:
        models.insert(0, current)
    await cl.ChatSettings(
        [Select(id="model", label="Modelo", values=models, initial_index=models.index(current))]
    ).send()


@cl.on_settings_update
async def on_settings_update(values: dict[str, Any]) -> None:
    agent = cl.user_session.get("agent")
    agent.model.model = values["model"]  # el historial se conserva; solo cambia quién responde
    await cl.Message(content=f"Modelo cambiado a `{values['model']}`.").send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    if message.content.strip().lower() == "/cyborg":  # reabrir el panel sin molestar al modelo
        await open_cyborg()
        return

    agent = cl.user_session.get("agent")
    waiting: Waiting = cl.user_session.get("waiting")
    events = agent.run(message.content)

    streaming: cl.Message | None = None
    answered: cl.Message | None = None  # la respuesta final, para colgarle las métricas
    try:
        await waiting.start(agent.model.model)
        while True:
            # Cada `next` corre en un hilo; así el bucle de eventos queda libre para los botones.
            event = await cl.make_async(next)(events, None)
            await waiting.stop()  # cualquier evento significa que el modelo ya respondió
            if event is None:
                break
            if event.kind == "token":
                streaming = streaming or cl.Message(content="")
                await streaming.stream_token(event.text)
                continue
            if event.kind == "stats":
                continue  # métricas por llamada; al final llega el resumen del turno
            if event.kind == "summary":
                await _show_summary(answered, event.args)
                continue
            if event.kind == "answer" and streaming is not None:
                streaming.content = event.text
                await streaming.update()
                answered, streaming = streaming, None
                continue
            if streaming is not None:  # texto previo a una herramienta: se cierra antes del paso
                await streaming.update()
                streaming = None
            if event.kind == "tool_result":
                async with cl.Step(name=event.name, type="tool") as step:
                    step.input = json.dumps(event.args, ensure_ascii=False)
                    step.output = event.text
                await waiting.start(agent.model.model)  # el modelo vuelve a pensar con el resultado
            elif event.kind == "answer":
                answered = cl.Message(content=event.text)
                await answered.send()
            elif event.kind == "notice":
                await cl.Message(content=f"ℹ️ {event.text}").send()
            elif event.kind == "error":
                await cl.Message(content=f"⚠️ {event.text}").send()
    except OllamaError as exc:
        await cl.Message(content=f"⚠️ {exc}").send()
    finally:
        await waiting.stop()


async def _show_summary(answered: cl.Message | None, summary: dict[str, Any]) -> None:
    """Chips de tiempos y contexto debajo de la respuesta: van aparte del texto, así no se copian con él."""
    element = cl.CustomElement(
        name="TurnMetrics",
        display="inline",
        props={"parts": [{"kind": k, "text": t} for k, t in summary_parts(summary)], "level": context_level(summary)},
    )
    if answered is None:  # el turno terminó sin respuesta (error o cortafuegos)
        await cl.Message(content="", elements=[element]).send()
    else:
        await element.send(for_id=answered.id)
