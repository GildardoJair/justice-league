"""Lecturas del hardware.

Concepto: *sondeo barato*. psutil lee /proc y NVML habla directo con el driver de NVIDIA
(lo mismo que usa `nvidia-smi`, pero sin lanzar un proceso). Cada lectura tarda
milisegundos y no toca los núcleos CUDA ni la VRAM, así que se puede consultar cada
pocos segundos sin frenar al modelo.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import psutil

from ..llm import running_models

try:
    import pynvml
except ImportError:  # sin nvidia-ml-py el resto sigue funcionando; la GPU sale como no disponible
    pynvml = None

T = TypeVar("T")

# Sensores de temperatura de CPU conocidos y la etiqueta preferida de cada uno.
CPU_SENSORS = {"k10temp": "Tctl", "zenpower": "Tdie", "coretemp": "Package id 0", "cpu_thermal": ""}

HWMON = Path("/sys/class/hwmon")

_nvml_ready: bool | None = None  # None: aún no se intentó iniciar NVML
_cpu_temp_file: Path | None | bool = False  # False: aún no se buscó; None: no hay

# psutil mide el uso de CPU entre dos llamadas; la primera siempre da 0. La hacemos aquí.
psutil.cpu_percent(interval=None)


def snapshot(ollama_host: str = "http://localhost:11434", cpu_interval: float | None = None) -> dict[str, Any]:
    """Foto del equipo en este instante. Los tamaños van en bytes; las temperaturas, en °C.

    `cpu_interval`: None mide el uso de CPU desde la consulta anterior (para sondeos
    periódicos, sin esperar); un número espera esos segundos y mide en ese tramo (consultas sueltas).
    """
    return {
        "time": time.time(),
        "gpu": gpu(),
        "cpu": cpu(cpu_interval),
        "ram": ram(),
        "ollama": ollama(ollama_host),
    }


def gpu() -> dict[str, Any] | None:
    """La primera GPU NVIDIA, o None si no hay (o falta el driver o nvidia-ml-py)."""
    if not _nvml():
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        return {
            "name": name.decode() if isinstance(name, bytes) else name,
            "util_percent": pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
            "mem_used": memory.used,
            "mem_total": memory.total,
            "temp_c": _maybe(lambda: pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)),
            "power_w": _maybe(lambda: pynvml.nvmlDeviceGetPowerUsage(handle) / 1000),
            "power_limit_w": _maybe(lambda: pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000),
        }
    except pynvml.NVMLError:
        return None


def cpu(interval: float | None = None) -> dict[str, Any]:
    freq = _maybe(psutil.cpu_freq)
    return {
        "percent": psutil.cpu_percent(interval=interval),
        "cores": psutil.cpu_count(logical=True),
        "freq_mhz": freq.current if freq else None,
        "temp_c": _cpu_temp(),
    }


def ram() -> dict[str, Any]:
    memory, swap = psutil.virtual_memory(), psutil.swap_memory()
    # "Usada" = lo que no está disponible. La caché de archivos de Linux (que guarda, por
    # ejemplo, los pesos de los modelos) cuenta como disponible: se libera si hace falta.
    return {
        "used": memory.total - memory.available,
        "total": memory.total,
        "swap_used": swap.used,
        "swap_total": swap.total,
    }


def ollama(host: str) -> list[dict[str, Any]] | None:
    """Modelos cargados y cómo se reparten entre VRAM y RAM; None si Ollama no responde."""
    models = running_models(host)
    if models is None:
        return None
    return [
        {
            "name": m.get("name", ""),
            "size": m.get("size", 0),
            "size_vram": m.get("size_vram", 0),
            "context_length": m.get("context_length"),
            "expires_at": m.get("expires_at"),
        }
        for m in models
    ]


def _cpu_temp() -> float | None:
    """Temperatura de la CPU. Lee directamente el archivo del sensor (Linux): leer todos los
    sensores con psutil costaba ~14 ms por consulta; un solo archivo, una fracción de eso."""
    global _cpu_temp_file
    if _cpu_temp_file is False:
        _cpu_temp_file = _find_cpu_temp_file()
    if _cpu_temp_file is None:
        return None
    raw = _maybe(_cpu_temp_file.read_text)
    return int(raw) / 1000 if raw else None


def _find_cpu_temp_file() -> Path | None:
    """Busca una vez el sensor de la CPU en /sys/class/hwmon y su lectura preferida."""
    for chip, label in CPU_SENSORS.items():
        for folder in sorted(HWMON.glob("hwmon*")):
            if _maybe((folder / "name").read_text) != f"{chip}\n":
                continue
            inputs = sorted(folder.glob("temp*_input"))
            for path in inputs:
                label_file = path.with_name(path.name.replace("_input", "_label"))
                if label and _maybe(label_file.read_text) == f"{label}\n":
                    return path
            if inputs:
                return inputs[0]
    return None


def _nvml() -> bool:
    global _nvml_ready
    if _nvml_ready is None:
        try:
            pynvml.nvmlInit()
            _nvml_ready = True
        except Exception:  # sin pynvml, sin driver o sin GPU NVIDIA
            _nvml_ready = False
    return _nvml_ready


def _maybe(read: Callable[[], T]) -> T | None:
    """Algunas lecturas no existen en todos los equipos (consumo, frecuencia, sensores)."""
    try:
        return read()
    except Exception:
        return None
