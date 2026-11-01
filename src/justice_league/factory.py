"""Ensamblado del agente. Único punto que conoce todas las piezas.

CLI y Atalaya llaman aquí en vez de construir el agente cada una por su lado:
así comparten exactamente el mismo comportamiento.
"""

from __future__ import annotations

from pathlib import Path

from .agent import Agent
from .config import Settings
from .llm import OllamaChat
from .permissions import Approver
from .tools import default_tools
from .zona_fantasma import Sandbox


def build_agent(
    settings: Settings,
    role_key: str,
    workspace: Path,
    approve: Approver,
    model_override: str | None = None,
) -> Agent:
    if role_key not in settings.roles:
        raise KeyError(f"Rol desconocido: {role_key}. Roles: {', '.join(settings.roles)}")
    role = settings.roles[role_key]

    model = OllamaChat(
        model=model_override or role.model,
        host=settings.ollama_host,
        num_ctx=role.num_ctx,
        temperature=role.temperature,
    )
    tools = default_tools(Sandbox(workspace))
    return Agent(model=model, tools=tools, approve=approve, system_prompt=role.system_prompt)
