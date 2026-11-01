"""Guardián de permisos (aún sin nombre de personaje).

Un *approver* es una función que recibe qué herramienta se quiere usar, con qué
argumentos y una vista previa legible (por ejemplo un diff), y responde sí o no.
La CLI lo implementa preguntando en la terminal; Atalaya, con botones en el chat.
El agente no sabe nada de eso: solo llama a la función.

Concepto: *inyección de dependencias*. El agente recibe el approver desde fuera
en vez de decidir por sí mismo cómo preguntar. Eso lo hace probable (en los tests
pasamos un approver falso) y reutilizable en cualquier interfaz.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# (nombre_herramienta, argumentos, vista_previa | None) -> aprobado
Approver = Callable[[str, dict[str, Any], str | None], bool]


def deny_all(name: str, args: dict[str, Any], preview: str | None) -> bool:
    return False


def approve_all(name: str, args: dict[str, Any], preview: str | None) -> bool:
    return True
