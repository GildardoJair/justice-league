"""Cyborg: diagnóstico del equipo (GPU, CPU, RAM y el modelo que Ollama tiene cargado).

Concepto: *una sola fuente de verdad*. `snapshot()` lee todo de una vez y devuelve un
diccionario simple (se puede pasar a JSON tal cual). No sabe de interfaces: Atalaya lo
dibuja en su panel lateral, `jl doctor` lo imprime y, más adelante, el agente podrá
consultarlo como herramienta para decidir qué modelo cargar.
"""

from .sensors import snapshot

__all__ = ["snapshot"]
