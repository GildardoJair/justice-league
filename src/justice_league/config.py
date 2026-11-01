"""Carga de roles desde roles.toml.

Concepto: *un rol es configuración, no código*. Superman, Batman y Mr. Terrific
pueden compartir el mismo bucle de agente; lo que cambia es el modelo, el prompt
de sistema y las herramientas. Por eso los roles viven en un archivo TOML que
puedes editar sin tocar Python.

Orden de búsqueda: variable JL_ROLES → ~/.config/justice_league/roles.toml →
el roles.toml que viene con el paquete.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROLES_PATH = Path(__file__).parent / "roles.toml"
USER_ROLES_PATH = Path.home() / ".config" / "justice_league" / "roles.toml"


@dataclass(frozen=True)
class Role:
    key: str
    name: str
    description: str
    model: str
    system_prompt: str
    enabled: bool = True
    num_ctx: int = 8192
    temperature: float = 0.2


@dataclass(frozen=True)
class Settings:
    ollama_host: str
    roles: dict[str, Role]


def find_roles_file() -> Path:
    if env := os.environ.get("JL_ROLES"):
        return Path(env).expanduser()
    if USER_ROLES_PATH.is_file():
        return USER_ROLES_PATH
    return DEFAULT_ROLES_PATH


def load_settings(path: Path | None = None) -> Settings:
    path = path or find_roles_file()
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    roles = {
        key: Role(
            key=key,
            name=raw.get("name", key.title()),
            description=raw.get("description", ""),
            model=raw["model"],
            system_prompt=raw.get("system_prompt", "").strip(),
            enabled=raw.get("enabled", True),
            num_ctx=raw.get("num_ctx", 8192),
            temperature=raw.get("temperature", 0.2),
        )
        for key, raw in data.get("roles", {}).items()
    }
    return Settings(ollama_host=data.get("ollama", {}).get("host", "http://localhost:11434"), roles=roles)
