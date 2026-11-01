import pytest

from justice_league.zona_fantasma import Sandbox, SandboxError


def test_resuelve_ruta_dentro_de_la_raiz(tmp_path):
    (tmp_path / "datos").mkdir()
    sandbox = Sandbox(tmp_path)
    assert sandbox.resolve("datos/a.csv") == tmp_path.resolve() / "datos" / "a.csv"


@pytest.mark.parametrize("ruta", ["../fuera.txt", "datos/../../fuera.txt", "/etc/passwd"])
def test_rechaza_rutas_que_escapan(tmp_path, ruta):
    with pytest.raises(SandboxError):
        Sandbox(tmp_path).resolve(ruta)


def test_rechaza_symlink_que_apunta_fuera(tmp_path):
    fuera = tmp_path / "fuera"
    fuera.mkdir()
    (fuera / "secreto.txt").write_text("x")
    proyecto = tmp_path / "proyecto"
    proyecto.mkdir()
    (proyecto / "atajo").symlink_to(fuera)

    with pytest.raises(SandboxError):
        Sandbox(proyecto).resolve("atajo/secreto.txt")


def test_git_es_solo_lectura(tmp_path):
    (tmp_path / ".git").mkdir()
    sandbox = Sandbox(tmp_path)
    sandbox.resolve(".git/config")  # leer se permite
    with pytest.raises(SandboxError):
        sandbox.resolve_for_write(".git/config")


def test_raiz_inexistente(tmp_path):
    with pytest.raises(SandboxError):
        Sandbox(tmp_path / "no-existe")
