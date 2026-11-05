from rich.console import Console

from justice_league.ui import banner, hardware_table, model_split, context_level, summary_line, summary_parts, summary_text


def resumen(**cambios):
    base = {"seconds": 12.34, "calls": 1, "load_seconds": 0.1, "first_token_seconds": 4.0,
            "eval_tokens": 400, "eval_seconds": 10.0, "context_tokens": 1000, "num_ctx": 8192}
    return base | cambios


def test_resumen_muestra_total_velocidad_y_contexto():
    texto = summary_text(resumen(load_seconds=5.7, calls=3))
    assert texto == "total 12.3 s · 40.0 tok/s · contexto 1,000/8,192 (12%)"


def test_resumen_sin_tiempo_ni_num_ctx_no_divide_por_cero():
    texto = summary_text(resumen(eval_seconds=0, num_ctx=0))
    assert "— tok/s" in texto and "contexto 1,000 tokens" in texto


def test_las_partes_llevan_su_tipo():
    assert [tipo for tipo, _ in summary_parts(resumen())] == ["total", "speed", "context"]


def test_nivel_de_contexto():
    assert context_level(resumen(context_tokens=1000)) == "ok"
    assert context_level(resumen(context_tokens=6000)) == "warn"
    assert context_level(resumen(context_tokens=7500)) == "full"
    assert summary_line(resumen(context_tokens=7500)).style == "red"


def test_banner_se_dibuja(tmp_path):
    consola = Console(record=True, width=80)
    consola.print(banner("Superman", "qwen2.5:14b", tmp_path, 8192))
    salida = consola.export_text()
    assert "Superman" in salida and "qwen2.5:14b" in salida


def foto(**cambios):
    base = {
        "time": 0,
        "gpu": {"name": "RTX 3060", "util_percent": 40, "mem_used": 6 * 2**30, "mem_total": 12 * 2**30,
                "temp_c": 85, "power_w": 120.0, "power_limit_w": 170.0},
        "cpu": {"percent": 12.0, "cores": 16, "freq_mhz": 3500.0, "temp_c": 50.0},
        "ram": {"used": 8 * 2**30, "total": 32 * 2**30, "swap_used": 0, "swap_total": 2**31},
        "ollama": [{"name": "qwen3:30b-a3b", "size": 100, "size_vram": 58, "context_length": 8192, "expires_at": None}],
    }
    return base | cambios


def test_tabla_de_hardware_se_dibuja():
    consola = Console(record=True, width=120)
    consola.print(hardware_table(foto()))
    salida = consola.export_text()
    assert "RTX 3060 · 40% uso · 85 °C · 120/170 W" in salida
    assert "6.0 GiB / 12.0 GiB (50%)" in salida and "42% CPU / 58% GPU" in salida


def test_tabla_sin_gpu_ni_ollama():
    consola = Console(record=True, width=120)
    consola.print(hardware_table(foto(gpu=None, ollama=None)))
    salida = consola.export_text()
    assert "no disponible" in salida and "no responde" in salida


def test_reparto_del_modelo():
    assert model_split({"size": 10, "size_vram": 10}) == "100% GPU"
    assert model_split({"size": 10, "size_vram": 0}) == "100% CPU"
