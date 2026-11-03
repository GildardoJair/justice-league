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
import json
import os
from pathlib import Path
from typing import Any

import chainlit as cl
from chainlit.input_widget import Select

from justice_league.config import load_settings
from justice_league.factory import build_agent
from justice_league.llm import OllamaError, list_models

ROLE = "superman"


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
    await cl.Message(content=f"**{settings.roles[ROLE].name}** listo en `{workspace}`.").send()

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
    agent = cl.user_session.get("agent")
    events = agent.run(message.content)

    streaming: cl.Message | None = None
    try:
        while True:
            # Cada `next` corre en un hilo; así el bucle de eventos queda libre para los botones.
            event = await cl.make_async(next)(events, None)
            if event is None:
                break
            if event.kind == "token":
                streaming = streaming or cl.Message(content="")
                await streaming.stream_token(event.text)
                continue
            if event.kind == "stats":
                continue  # métricas: la web aún no las muestra
            if event.kind == "answer" and streaming is not None:
                streaming.content = event.text
                await streaming.update()
                streaming = None
                continue
            if streaming is not None:  # texto previo a una herramienta: se cierra antes del paso
                await streaming.update()
                streaming = None
            if event.kind == "tool_result":
                async with cl.Step(name=event.name, type="tool") as step:
                    step.input = json.dumps(event.args, ensure_ascii=False)
                    step.output = event.text
            elif event.kind == "answer":
                await cl.Message(content=event.text).send()
            elif event.kind == "error":
                await cl.Message(content=f"⚠️ {event.text}").send()
    except OllamaError as exc:
        await cl.Message(content=f"⚠️ {exc}").send()
