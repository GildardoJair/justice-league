"""Piezas visuales de la terminal, separadas de la lógica de `cli.py`.

Concepto: *presentación aparte de comportamiento*. Aquí solo hay texto y colores;
nada de esto sabe cómo funciona el agente.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Group
from rich.panel import Panel
from rich.text import Text

LOGO = "  ⚡ J U S T I C E   L E A G U E"


def banner(role_name: str, model: str, workspace: Path, num_ctx: int) -> Panel:
    info = Text.assemble(
        ("  Rol       ", "dim"), (f"{role_name}\n", "bold"),
        ("  Modelo    ", "dim"), (f"{model}", "cyan"), (f"  (contexto {num_ctx:,})\n", "dim"),
        ("  Carpeta   ", "dim"), (f"{workspace}\n\n", ""),
        ("  /reset", "bold"), (" reinicia · ", "dim"), ("/salir", "bold"), (" termina", "dim"),
    )
    return Panel(Group(Text(LOGO, style="bold blue"), Text(""), info), border_style="blue")


def stats_line(stats: dict[str, Any], num_ctx: int) -> Text:
    """Resumen de una vuelta: velocidad y cuánto del contexto se usa."""
    tokens, seconds = stats["eval_tokens"], stats["eval_seconds"]
    used = stats["prompt_tokens"] + tokens
    speed = f"{tokens / seconds:.1f} tok/s" if seconds else "— tok/s"
    pct = used / num_ctx * 100
    style = "red" if pct >= 90 else "yellow" if pct >= 70 else "dim"
    return Text(f"  {speed} · {tokens} tokens generados · contexto {used:,}/{num_ctx:,} ({pct:.0f}%)", style=style)
