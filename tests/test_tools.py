import pytest

from justice_league.tools import default_tools
from justice_league.zona_fantasma import Sandbox, SandboxError


@pytest.fixture
def tools(tmp_path):
    return default_tools(Sandbox(tmp_path))


def test_las_escrituras_se_preparan_y_no_tocan_el_disco(tools, tmp_path):
    assert tools.get("write_file").fn("notas/a.md", "hola").startswith("OK")
    assert not (tmp_path / "notas").exists()
    assert tools.staging.pending() and tools.staging.names() == ["notas/a.md"]


def test_el_modelo_ve_lo_preparado_y_apply_escribe(tools, tmp_path):
    tools.get("write_file").fn("notas/a.md", "hola")
    assert tools.get("read_file").fn("notas/a.md") == "hola"  # lee su versión nueva
    assert "notas/" in tools.get("list_dir").fn(".")
    assert tools.get("list_dir").fn("notas") == "a.md"
    tools.staging.apply()
    assert (tmp_path / "notas" / "a.md").read_text() == "hola"
    assert not tools.staging.pending()


def test_discard_no_deja_rastro(tools, tmp_path):
    tools.get("write_file").fn("a.txt", "x")
    tools.staging.discard()
    assert not (tmp_path / "a.txt").exists() and tools.get("read_file").fn("a.txt").startswith("ERROR")


def test_varias_ediciones_al_mismo_archivo_dan_un_solo_diff(tools, tmp_path):
    (tmp_path / "a.txt").write_text("uno\ndos\ntres\n")
    edit = tools.get("edit_file").fn
    edit("a.txt", "uno", "UNO")
    edit("a.txt", "tres", "TRES")  # parte de la versión ya editada
    diff = tools.staging.diff()
    assert diff.count("--- a.txt") == 1 and "+UNO" in diff and "+TRES" in diff
    assert (tmp_path / "a.txt").read_text() == "uno\ndos\ntres\n"  # disco intacto hasta apply


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
    tools.staging.apply()
    assert (tmp_path / "a.py").read_text() == "x = 9\ny = 1\n"


def test_el_diff_muestra_antes_y_despues(tools, tmp_path):
    (tmp_path / "a.txt").write_text("uno\n")
    tools.get("write_file").fn("a.txt", "dos\n")
    diff = tools.staging.diff()
    assert "-uno" in diff and "+dos" in diff


def test_editar_un_archivo_inexistente_da_error(tools):
    assert tools.get("edit_file").fn("no.txt", "a", "b").startswith("ERROR")


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


def test_edit_que_rompe_el_py_avisa_y_el_diff_final_tambien(tools, tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    assert "AVISO" in tools.get("edit_file").fn("a.py", "x = 1", "x = (")
    assert "⚠️" in tools.staging.diff()


def test_diff_sin_error_no_lleva_advertencia(tools):
    tools.get("write_file").fn("a.py", "x = 1\n")
    assert "⚠️" not in tools.staging.diff()
