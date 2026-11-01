"""Herramientas que el agente puede usar y el registro donde viven.

Cada herramienta es: un nombre, una descripción para el modelo, un esquema JSON de
sus parámetros, una función Python y una bandera `needs_approval`.

Concepto: *tool calling*. El modelo no ejecuta nada; devuelve "quiero llamar a
read_file con path=x". Nuestro código es quien ejecuta, y decide si pedir permiso.
Las descripciones y esquemas son el "manual" que el modelo lee para elegir bien,
así que su redacción importa tanto como el código.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .zona_fantasma import Sandbox

MAX_READ_CHARS = 20_000


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., str]
    needs_approval: bool = False
    preview: Callable[..., str] | None = None  # vista previa (diff) para el guardián

    def spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)


def _diff(old: str, new: str, label: str) -> str:
    lines = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"{label} (antes)",
        tofile=f"{label} (después)",
    )
    return "".join(lines) or "(sin cambios)"


def default_tools(sandbox: Sandbox) -> ToolRegistry:
    """Herramientas de archivos de la Fase 0. Todo pasa por el sandbox."""

    def list_dir(path: str = ".") -> str:
        target = sandbox.resolve(path)
        if not target.is_dir():
            return f"ERROR: {path} no es una carpeta"
        entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        return "\n".join(f"{p.name}/" if p.is_dir() else p.name for p in entries) or "(carpeta vacía)"

    def read_file(path: str) -> str:
        target = sandbox.resolve(path)
        if not target.is_file():
            return f"ERROR: {path} no es un archivo"
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"ERROR: {path} no es texto UTF-8 (¿binario?)"
        if len(text) > MAX_READ_CHARS:
            return text[:MAX_READ_CHARS] + f"\n\n[... truncado: {len(text)} caracteres en total ...]"
        return text

    def write_file(path: str, content: str) -> str:
        target = sandbox.resolve_for_write(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"OK: escrito {sandbox.display(target)} ({len(content)} caracteres)"

    def write_preview(path: str, content: str) -> str:
        target = sandbox.resolve_for_write(path)
        old = target.read_text(encoding="utf-8") if target.is_file() else ""
        return _diff(old, content, sandbox.display(target))

    def edit_file(path: str, old: str, new: str) -> str:
        target = sandbox.resolve_for_write(path)
        text = target.read_text(encoding="utf-8")
        count = text.count(old)
        if count == 0:
            return "ERROR: el texto `old` no aparece en el archivo"
        if count > 1:
            return f"ERROR: `old` aparece {count} veces; incluye más contexto para que sea único"
        target.write_text(text.replace(old, new, 1), encoding="utf-8")
        return f"OK: editado {sandbox.display(target)}"

    def edit_preview(path: str, old: str, new: str) -> str:
        target = sandbox.resolve_for_write(path)
        text = target.read_text(encoding="utf-8")
        return _diff(text, text.replace(old, new, 1), sandbox.display(target))

    def obj(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
        return {"type": "object", "properties": properties, "required": required}

    string = {"type": "string"}
    return ToolRegistry(
        [
            Tool(
                "list_dir",
                "Lista los archivos y carpetas de una carpeta del proyecto.",
                obj({"path": {**string, "description": "Carpeta relativa al proyecto. Por defecto '.'"}}, []),
                list_dir,
            ),
            Tool(
                "read_file",
                "Lee un archivo de texto del proyecto y devuelve su contenido.",
                obj({"path": {**string, "description": "Ruta relativa al proyecto"}}, ["path"]),
                read_file,
            ),
            Tool(
                "write_file",
                "Crea un archivo nuevo o reemplaza por completo uno existente. Requiere aprobación del usuario.",
                obj(
                    {"path": {**string, "description": "Ruta relativa al proyecto"}, "content": {**string, "description": "Contenido completo"}},
                    ["path", "content"],
                ),
                write_file,
                needs_approval=True,
                preview=write_preview,
            ),
            Tool(
                "edit_file",
                "Reemplaza un fragmento exacto de un archivo existente. `old` debe aparecer una sola vez. "
                "Preferible a write_file para cambios pequeños. Requiere aprobación del usuario.",
                obj(
                    {
                        "path": {**string, "description": "Ruta relativa al proyecto"},
                        "old": {**string, "description": "Texto exacto a reemplazar (único en el archivo)"},
                        "new": {**string, "description": "Texto nuevo"},
                    },
                    ["path", "old", "new"],
                ),
                edit_file,
                needs_approval=True,
                preview=edit_preview,
            ),
        ]
    )
