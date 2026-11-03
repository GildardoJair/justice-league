from rich.console import Console

from justice_league.ui import banner, stats_line


def test_stats_line_muestra_velocidad_y_contexto():
    texto = stats_line({"prompt_tokens": 600, "eval_tokens": 400, "eval_seconds": 10.0}, 8192).plain
    assert "40.0 tok/s" in texto and "1,000/8,192" in texto and "(12%)" in texto


def test_stats_line_sin_tiempo_no_divide_por_cero():
    assert "— tok/s" in stats_line({"prompt_tokens": 1, "eval_tokens": 1, "eval_seconds": 0}, 100).plain


def test_banner_se_dibuja(tmp_path):
    consola = Console(record=True, width=80)
    consola.print(banner("Superman", "qwen2.5:14b", tmp_path, 8192))
    salida = consola.export_text()
    assert "Superman" in salida and "qwen2.5:14b" in salida
