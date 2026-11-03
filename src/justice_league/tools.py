"""Herramientas que el agente puede usar y el registro donde viven.

Cada herramienta es: un nombre, una descripción para el modelo, un esquema JSON de
sus parámetros, una función Python y una bandera `needs_approval`.

Concepto: *tool calling*. El modelo no ejecuta nada; devuelve "quiero llamar a
read_file con path=x". Nuestro código es quien ejecuta, y decide si pedir permiso.
Las descripciones y esquemas son el "manual" que el modelo lee para elegir bien,
así que su redacción importa tanto como el código.

Las escrituras no tocan el disco: pasan por `Staging` y el agente pide una sola
aprobación (un diff combinado) al terminar el turno.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
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
        self.staging: Staging | None = None  # lo asigna default_tools; el agente lo consulta al final del turno
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


def _syntax_error(path: str, text: str) -> str | None:
    """Si `path` es un .py con sintaxis inválida, describe el error. `compile` solo analiza: no ejecuta nada."""
    if not path.endswith(".py"):
        return None
    try:
        compile(text, path, "exec")
    except SyntaxError as exc:
        return f"línea {exc.lineno}: {exc.msg}"
    return None


def _syntax_warning(path: str, text: str) -> str:
    error = _syntax_error(path, text)
    return f"\nAVISO: el archivo quedó con un error de sintaxis ({error}). Léelo y corrígelo antes de seguir." if error else ""


def _syntax_note(path: str, text: str) -> str:
    error = _syntax_error(path, text)
    return f"\n⚠️ Con este cambio el archivo tendría un error de sintaxis ({error})." if error else ""


class Staging:
    """Área de preparación: las escrituras del modelo viven aquí hasta que el usuario las aprueba.

    Concepto: *staging* (como el `git add` previo al commit). El modelo lee y edita su versión
    nueva, pero el disco no cambia hasta `apply()`. Así un solo diff final sustituye a una
    aprobación por edición, sin romper la regla de que nada se escribe sin tu sí.
    """

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox
        self._files: dict[Path, list[str | None]] = {}  # ruta -> [texto original (None si no existía), texto nuevo]

    def current_text(self, target: Path) -> str | None:
        if target in self._files:
            return self._files[target][1]
        return target.read_text(encoding="utf-8") if target.is_file() else None

    def write(self, target: Path, text: str) -> None:
        if target in self._files:
            self._files[target][1] = text
        else:
            original = target.read_text(encoding="utf-8") if target.is_file() else None
            self._files[target] = [original, text]

    def _changed(self) -> dict[Path, list[str | None]]:
        return {t: pair for t, pair in self._files.items() if pair[0] != pair[1]}

    def pending(self) -> bool:
        return bool(self._changed())

    def names(self) -> list[str]:
        return [self.sandbox.display(t) for t in self._changed()]

    def new_children(self, folder: Path) -> list[tuple[str, bool]]:
        """Entradas de `folder` que solo existen en el área de preparación: (nombre, es_carpeta)."""
        found: dict[str, bool] = {}
        for target in self._changed():
            if target.exists():
                continue
            try:
                parts = target.relative_to(folder).parts
            except ValueError:
                continue
            found.setdefault(parts[0], len(parts) > 1)
        return list(found.items())

    def diff(self) -> str:
        blocks = []
        for target, (old, new) in self._changed().items():
            label = self.sandbox.display(target)
            blocks.append(_diff(old or "", new or "", label) + _syntax_note(label, new or ""))
        return "\n".join(blocks)

    def apply(self) -> None:
        for target, (_, new) in self._changed().items():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(new or "", encoding="utf-8")
        self._files.clear()

    def discard(self) -> None:
        self._files.clear()


def default_tools(sandbox: Sandbox) -> ToolRegistry:
    """Herramientas de archivos de la Fase 0. Todo pasa por el sandbox; las escrituras, por el área de preparación."""
    staging = Staging(sandbox)

    def list_dir(path: str = ".") -> str:
        target = sandbox.resolve(path)
        staged = staging.new_children(target)
        if not target.is_dir() and not staged:
            return f"ERROR: {path} no es una carpeta"
        entries = {p.name: p.is_dir() for p in target.iterdir()} if target.is_dir() else {}
        for name, is_dir in staged:
            entries.setdefault(name, is_dir)
        ordered = sorted(entries.items(), key=lambda kv: (not kv[1], kv[0].lower()))
        return "\n".join(f"{name}/" if is_dir else name for name, is_dir in ordered) or "(carpeta vacía)"

    def read_file(path: str) -> str:
        target = sandbox.resolve(path)
        try:
            text = staging.current_text(target)
        except UnicodeDecodeError:
            return f"ERROR: {path} no es texto UTF-8 (¿binario?)"
        if text is None:
            return f"ERROR: {path} no es un archivo"
        if len(text) > MAX_READ_CHARS:
            return text[:MAX_READ_CHARS] + f"\n\n[... truncado: {len(text)} caracteres en total ...]"
        return text

    def write_file(path: str, content: str) -> str:
        target = sandbox.resolve_for_write(path)
        staging.write(target, content)
        return f"OK: preparado {sandbox.display(target)} ({len(content)} caracteres); se escribe cuando el usuario apruebe el diff final" + _syntax_warning(path, content)

    def edit_file(path: str, old: str, new: str) -> str:
        target = sandbox.resolve_for_write(path)
        text = staging.current_text(target)
        if text is None:
            return f"ERROR: {path} no existe"
        count = text.count(old)
        if count == 0:
            return "ERROR: el texto `old` no aparece en el archivo"
        if count > 1:
            return f"ERROR: `old` aparece {count} veces; incluye más contexto para que sea único"
        result = text.replace(old, new, 1)
        staging.write(target, result)
        return f"OK: cambio preparado en {sandbox.display(target)}; se escribe cuando el usuario apruebe el diff final" + _syntax_warning(path, result)

    def obj(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
        return {"type": "object", "properties": properties, "required": required}

    string = {"type": "string"}
    registry = ToolRegistry(
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
                "Crea un archivo nuevo o reemplaza por completo uno existente. El cambio queda preparado y el usuario lo aprueba al final.",
                obj(
                    {"path": {**string, "description": "Ruta relativa al proyecto"}, "content": {**string, "description": "Contenido completo"}},
                    ["path", "content"],
                ),
                write_file,
            ),
            Tool(
                "edit_file",
                "Reemplaza un fragmento exacto de un archivo existente. `old` debe aparecer una sola vez. "
                "Preferible a write_file para cambios pequeños. El cambio queda preparado y el usuario lo aprueba al final.",
                obj(
                    {
                        "path": {**string, "description": "Ruta relativa al proyecto"},
                        "old": {**string, "description": "Texto exacto a reemplazar (único en el archivo)"},
                        "new": {**string, "description": "Texto nuevo"},
                    },
                    ["path", "old", "new"],
                ),
                edit_file,
            ),
        ]
    )
    registry.staging = staging
    return registry
