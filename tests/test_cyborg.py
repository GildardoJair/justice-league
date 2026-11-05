"""Pruebas de Cyborg con el hardware simulado: no hace falta GPU ni Ollama."""

from justice_league.cyborg import sensors, snapshot


def test_snapshot_trae_todas_las_secciones(monkeypatch):
    monkeypatch.setattr(sensors, "running_models", lambda host: [])
    foto = snapshot()
    assert set(foto) == {"time", "gpu", "cpu", "ram", "ollama"}
    assert foto["cpu"]["cores"] >= 1 and 0 <= foto["ram"]["used"] <= foto["ram"]["total"]
    assert foto["ollama"] == []


def test_sin_nvml_la_gpu_sale_como_no_disponible(monkeypatch):
    monkeypatch.setattr(sensors, "_nvml", lambda: False)
    assert sensors.gpu() is None


def test_ollama_apagado_da_none(monkeypatch):
    monkeypatch.setattr(sensors, "running_models", lambda host: None)
    assert sensors.ollama("http://x") is None


def test_ollama_resume_cada_modelo(monkeypatch):
    crudo = [{"name": "qwen2.5:14b", "size": 10, "size_vram": 6, "context_length": 8192, "expires_at": "x", "digest": "..."}]
    monkeypatch.setattr(sensors, "running_models", lambda host: crudo)
    assert sensors.ollama("http://x") == [
        {"name": "qwen2.5:14b", "size": 10, "size_vram": 6, "context_length": 8192, "expires_at": "x"}
    ]


def test_encuentra_el_sensor_de_cpu_con_su_etiqueta(tmp_path, monkeypatch):
    nvme = tmp_path / "hwmon0"
    nvme.mkdir()
    (nvme / "name").write_text("nvme\n")
    (nvme / "temp1_input").write_text("39850\n")
    amd = tmp_path / "hwmon1"
    amd.mkdir()
    (amd / "name").write_text("k10temp\n")
    (amd / "temp1_input").write_text("50000\n")
    (amd / "temp1_label").write_text("Tccd1\n")
    (amd / "temp2_input").write_text("45875\n")
    (amd / "temp2_label").write_text("Tctl\n")
    monkeypatch.setattr(sensors, "HWMON", tmp_path)
    monkeypatch.setattr(sensors, "_cpu_temp_file", False)
    assert sensors._cpu_temp() == 45.875


def test_sin_sensor_de_cpu_la_temperatura_es_none(tmp_path, monkeypatch):
    monkeypatch.setattr(sensors, "HWMON", tmp_path)
    monkeypatch.setattr(sensors, "_cpu_temp_file", False)
    assert sensors._cpu_temp() is None
