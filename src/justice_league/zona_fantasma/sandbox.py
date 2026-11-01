"""Sandbox de archivos: el agente solo puede ver y tocar dentro de una carpeta raíz.

Concepto: *path traversal*. Un modelo puede pedir "../../.ssh/id_rsa" o seguir un
enlace simbólico que apunta fuera de la carpeta. Por eso nunca confiamos en la
ruta tal como llega: la resolvemos a su forma absoluta real (siguiendo symlinks)
y comprobamos que siga viviendo bajo la raíz.

Esto protege contra errores del modelo, no contra código malicioso arbitrario.
Cuando exista un ejecutor de código (Fase 4), necesitará aislamiento de verdad
(contenedor, usuario sin privilegios), no solo validación de rutas.
"""

from __future__ import annotations

from pathlib import Path

# Carpetas donde el agente puede leer pero nunca escribir.
PROTECTED_DIRS = {".git"}


class SandboxError(Exception):
    """La operación pedida sale del área permitida."""


class Sandbox:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise SandboxError(f"La carpeta de trabajo no existe: {self.root}")

    def resolve(self, path: str | Path) -> Path:
        """Convierte una ruta relativa (o absoluta) en una ruta real dentro de la raíz."""
        # Si `path` es absoluta, `root / path` la deja tal cual, y la comprobación
        # de abajo la rechaza si está fuera. Por eso no hay que tratarla aparte.
        candidate = (self.root / path).resolve()
        if not candidate.is_relative_to(self.root):
            raise SandboxError(f"Ruta fuera del espacio de trabajo: {path}")
        return candidate

    def resolve_for_write(self, path: str | Path) -> Path:
        """Igual que resolve(), pero además rechaza carpetas protegidas."""
        candidate = self.resolve(path)
        relative_parts = candidate.relative_to(self.root).parts
        if PROTECTED_DIRS.intersection(relative_parts):
            raise SandboxError(f"Ruta protegida (solo lectura): {path}")
        return candidate

    def display(self, path: Path) -> str:
        """Ruta relativa a la raíz, para mostrársela al modelo y al usuario."""
        try:
            return str(path.relative_to(self.root)) or "."
        except ValueError:
            return str(path)
