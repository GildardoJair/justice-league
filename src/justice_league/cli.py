"""Cara de terminal: `jl chat`, `jl roles`, `jl doctor`, `jl atalaya`.

Adaptador delgado: solo traduce entre la terminal y el núcleo (Agent).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from .config import find_roles_file, load_settings
from .factory import build_agent
from .llm import OllamaError
from .zona_fantasma import SandboxError

app = typer.Typer(help="Justice League: equipo local de agentes de IA.", no_args_is_help=True)
console = Console()


def _approve_in_terminal(name: str, args: dict[str, Any], preview: str | None) -> bool:
    body = Syntax(preview, "diff", theme="ansi_dark") if preview else str(args)
    console.print(Panel(body, title=f"🛡️  Permiso solicitado: {name}", border_style="yellow"))
    return typer.confirm("¿Aprobar?", default=False)


@app.command()
def chat(
    workspace: Annotated[Path, typer.Option("--workspace", "-w", help="Carpeta del proyecto en la que trabaja el agente.")] = Path("."),
    role: Annotated[str, typer.Option("--role", "-r", help="Rol a usar.")] = "superman",
    model: Annotated[str | None, typer.Option("--model", "-m", help="Sobrescribe el modelo del rol (p. ej. qwen2.5:7b-instruct).")] = None,
) -> None:
    """Chat interactivo con el agente en la carpeta indicada."""
    settings = load_settings()
    try:
        agent = build_agent(settings, role, workspace.resolve(), _approve_in_terminal, model)
    except (SandboxError, KeyError) as exc:
        console.print(f"[red]{exc.args[0]}[/]")
        raise typer.Exit(1)
    active = settings.roles[role]
    console.print(
        Panel(
            f"[bold]{active.name}[/] · modelo [cyan]{model or active.model}[/]\n"
            f"Carpeta: {workspace.resolve()}\n"
            "Comandos: [bold]/reset[/] reinicia la conversación · [bold]/salir[/] termina",
            title="Justice League",
        )
    )

    while True:
        try:
            text = console.input("[bold green]tú ›[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not text:
            continue
        if text in {"/salir", "/exit", "/q"}:
            break
        if text == "/reset":
            agent.reset()
            console.print("[dim]Conversación reiniciada.[/]")
            continue

        live: Live | None = None
        buffer = ""
        try:
            for event in agent.run(text):
                if event.kind == "token":
                    if live is None:
                        live = Live(console=console, refresh_per_second=12, vertical_overflow="visible")
                        live.start()
                    buffer += event.text
                    live.update(Panel(Markdown(buffer), title=active.name, border_style="blue"))
                    continue
                if live is not None:  # cualquier otro evento cierra el panel en vivo
                    live.stop()
                    live, buffer = None, ""
                    if event.kind == "answer":
                        continue  # ya se mostró token a token
                if event.kind == "tool_call":
                    console.print(f"[dim]🔧 {event.name}({event.args})[/]")
                elif event.kind == "tool_result":
                    shown = event.text if len(event.text) < 300 else event.text[:300] + " …"
                    console.print(f"[dim]   ↳ {shown}[/]")
                elif event.kind == "answer":
                    console.print(Panel(Markdown(event.text), title=active.name, border_style="blue"))
                else:
                    console.print(f"[red]{event.text}[/]")
        except OllamaError as exc:
            console.print(f"[red]{exc}[/]")
        finally:
            if live is not None:
                live.stop()


@app.command()
def roles() -> None:
    """Lista los roles configurados."""
    settings = load_settings()
    table = Table(title=f"Roles ({find_roles_file()})")
    for column in ("Clave", "Nombre", "Modelo", "Activo", "Descripción"):
        table.add_column(column)
    for r in settings.roles.values():
        table.add_row(r.key, r.name, r.model, "sí" if r.enabled else "no", r.description)
    console.print(table)


@app.command()
def doctor() -> None:
    """Comprueba que Ollama responde y que los modelos de los roles activos están descargados."""
    settings = load_settings()
    try:
        response = httpx.get(f"{settings.ollama_host}/api/tags", timeout=5)
        response.raise_for_status()
    except httpx.HTTPError:
        console.print(f"[red]✗ No hay respuesta de Ollama en {settings.ollama_host}. Prueba `ollama serve`.[/]")
        raise typer.Exit(1)

    console.print(f"[green]✓ Ollama responde en {settings.ollama_host}[/]")
    installed = {m["name"] for m in response.json().get("models", [])}
    missing = False
    for r in settings.roles.values():
        if not r.enabled:
            continue
        # Ollama usa ":latest" implícito cuando el nombre no trae etiqueta.
        present = r.model in installed or f"{r.model}:latest" in installed
        console.print(f"{'[green]✓' if present else '[red]✗'} {r.name}: {r.model}[/]" + ("" if present else f"  →  ollama pull {r.model}"))
        missing = missing or not present
    if missing:
        raise typer.Exit(1)


@app.command()
def atalaya(
    workspace: Annotated[Path, typer.Option("--workspace", "-w", help="Carpeta del proyecto.")] = Path("."),
    port: Annotated[int, typer.Option(help="Puerto de la interfaz web.")] = 8000,
) -> None:
    """Abre Atalaya, la interfaz web (requiere: uv sync --extra atalaya)."""
    try:
        import chainlit  # noqa: F401
    except ImportError:
        console.print("[red]Falta Chainlit. Instálalo con: uv sync --extra atalaya[/]")
        raise typer.Exit(1)

    entry = Path(__file__).parent / "atalaya" / "app.py"
    env = {**os.environ, "JL_WORKSPACE": str(workspace.resolve())}
    subprocess.run([sys.executable, "-m", "chainlit", "run", str(entry), "--port", str(port)], env=env, check=False)
