"""Piezas visuales de la terminal, separadas de la lógica de `cli.py`.

Concepto: *presentación aparte de comportamiento*. Aquí solo hay texto y colores;
nada de esto sabe cómo funciona el agente.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
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


def context_level(summary: dict[str, Any]) -> str:
    """"ok", "warn" (≥70 %) o "full" (≥90 %) según cuánto del contexto se usa."""
    num_ctx = summary.get("num_ctx") or 0
    pct = summary["context_tokens"] / num_ctx * 100 if num_ctx else 0
    return "full" if pct >= 90 else "warn" if pct >= 70 else "ok"


def summary_parts(summary: dict[str, Any]) -> list[tuple[str, str]]:
    """Las piezas visibles del resumen de un turno como (tipo, texto). Atalaya las dibuja como chips.

    El resumen trae más cifras (carga, primer token, llamadas); por ahora solo se muestran estas tres.
    """
    tokens, seconds = summary["eval_tokens"], summary["eval_seconds"]
    used, num_ctx = summary["context_tokens"], summary.get("num_ctx") or 0
    context = f"contexto {used:,}/{num_ctx:,} ({used / num_ctx * 100:.0f}%)" if num_ctx else f"contexto {used:,} tokens"
    return [
        ("total", f"total {summary['seconds']:.1f} s"),
        ("speed", f"{tokens / seconds:.1f} tok/s" if seconds else "— tok/s"),
        ("context", context),
    ]


def summary_text(summary: dict[str, Any]) -> str:
    """El resumen en una línea de texto plano (la terminal)."""
    return " · ".join(text for _, text in summary_parts(summary))


def summary_line(summary: dict[str, Any]) -> Text:
    """La misma línea, coloreada para la terminal según el contexto usado."""
    style = {"full": "red", "warn": "yellow"}.get(context_level(summary), "dim")
    return Text(f"  {summary_text(summary)}", style=style)


def _gib(n: float) -> str:
    return f"{n / 2**30:.1f} GiB"


def _temp(celsius: float | None, warn: float, hot: float) -> Text:
    if celsius is None:
        return Text("")
    return Text(f" · {celsius:.0f} °C", style="red" if celsius >= hot else "yellow" if celsius >= warn else "")


def hardware_table(snap: dict[str, Any]) -> Table:
    """La foto de Cyborg como tabla para `jl doctor`."""
    table = Table(title="Cyborg · estado del equipo", show_header=False, title_justify="left")
    table.add_column(style="dim")
    table.add_column()

    gpu = snap["gpu"]
    if gpu:
        power = f" · {gpu['power_w']:.0f}/{gpu['power_limit_w']:.0f} W" if gpu["power_w"] is not None and gpu["power_limit_w"] else ""
        table.add_row("GPU", Text.assemble(f"{gpu['name']} · {gpu['util_percent']}% uso", _temp(gpu["temp_c"], 80, 87), power))
        table.add_row("VRAM", f"{_gib(gpu['mem_used'])} / {_gib(gpu['mem_total'])} ({gpu['mem_used'] / gpu['mem_total'] * 100:.0f}%)")
    else:
        table.add_row("GPU", Text("no disponible (sin GPU NVIDIA o sin driver)", style="yellow"))

    cpu = snap["cpu"]
    table.add_row("CPU", Text.assemble(f"{cpu['percent']:.0f}% uso · {cpu['cores']} hilos", _temp(cpu["temp_c"], 80, 90)))
    ram = snap["ram"]
    swap = f" · swap {_gib(ram['swap_used'])}" if ram["swap_used"] > 2**20 else ""
    table.add_row("RAM", f"{_gib(ram['used'])} / {_gib(ram['total'])} ({ram['used'] / ram['total'] * 100:.0f}%){swap}")

    models = snap["ollama"]
    if models is None:
        table.add_row("Ollama", Text("no responde", style="red"))
    elif not models:
        table.add_row("Ollama", Text("ningún modelo cargado", style="dim"))
    for m in models or []:
        table.add_row("Ollama", f"{m['name']} · {_gib(m['size'])} · {model_split(m)}" + (f" · contexto {m['context_length']:,}" if m["context_length"] else ""))
    return table


def model_split(model: dict[str, Any]) -> str:
    """Reparto de un modelo cargado, como lo dice `ollama ps`: "100% GPU" o "42% CPU / 58% GPU"."""
    if not model["size"]:
        return "—"
    gpu = round(model["size_vram"] / model["size"] * 100)
    return "100% GPU" if gpu >= 100 else "100% CPU" if gpu <= 0 else f"{100 - gpu}% CPU / {gpu}% GPU"
