import pytest

from justice_league.tools import default_tools
from justice_league.zona_fantasma import Sandbox, SandboxError


@pytest.fixture
def tools(tmp_path):
    return default_tools(Sandbox(tmp_path))


def test_solo_las_escrituras_piden_aprobacion(tools):
    pide = {name: tools.get(name).needs_approval for name in tools.names()}
    assert pide == {"list_dir": False, "read_file": False, "write_file": True, "edit_file": True}


def test_write_y_read(tools, tmp_path):
    tools.get("write_file").fn("notas/a.md", "hola")
    assert tools.get("read_file").fn("notas/a.md") == "hola"
    assert (tmp_path / "notas" / "a.md").read_text() == "hola"


def test_read_trunca_archivos_grandes(tools, tmp_path):
    (tmp_path / "grande.txt").write_text("x" * 50_000)
    resultado = tools.get("read_file").fn("grande.txt")
    assert "truncado" in resultado and len(resultado) < 21_000


def test_edit_exige_coincidencia_unica(tools, tmp_path):
    (tmp_path / "a.py").write_text("x = 1\ny = 1\n")
    edit = tools.get("edit_file").fn
    assert "aparece 2 veces" in edit("a.py", "= 1", "= 2")
    assert "no aparece" in edit("a.py", "z", "w")
    assert edit("a.py", "x = 1", "x = 9").startswith("OK")
    assert (tmp_path / "a.py").read_text() == "x = 9\ny = 1\n"


def test_preview_muestra_diff(tools, tmp_path):
    (tmp_path / "a.txt").write_text("uno\n")
    diff = tools.get("write_file").preview("a.txt", "dos\n")
    assert "-uno" in diff and "+dos" in diff


def test_las_herramientas_respetan_el_sandbox(tools):
    with pytest.raises(SandboxError):
        tools.get("read_file").fn("../../etc/passwd")
    with pytest.raises(SandboxError):
        tools.get("write_file").fn("/tmp/fuera.txt", "x")


def test_write_py_con_sintaxis_invalida_avisa(tools):
    resultado = tools.get("write_file").fn("a.py", "def (:\n")
    assert resultado.startswith("OK") and "AVISO" in resultado and "sintaxis" in resultado


def test_write_py_valido_no_avisa(tools):
    assert "AVISO" not in tools.get("write_file").fn("a.py", "x = 1\n")


def test_solo_valida_archivos_py(tools):
    assert "AVISO" not in tools.get("write_file").fn("a.txt", "def (:\n")


def test_edit_que_rompe_el_py_avisa_y_el_preview_tambien(tools, tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    edit = tools.get("edit_file")
    assert "⚠️" in edit.preview("a.py", "x = 1", "x = (")
    assert "AVISO" in edit.fn("a.py", "x = 1", "x = (")


def test_preview_sin_error_no_lleva_advertencia(tools):
    assert "⚠️" not in tools.get("write_file").preview("a.py", "x = 1\n")
