"""Pruebas del bucle del agente con un modelo falso: sin Ollama, sin GPU, en milisegundos.

Concepto: *test double*. FakeModel responde lo que le programamos, así probamos la
lógica del bucle (herramientas, permisos, errores) de forma determinista.
"""

import pytest

from justice_league.agent import Agent
from justice_league.llm import ModelReply, ToolCall
from justice_league.permissions import approve_all, deny_all
from justice_league.tools import default_tools
from justice_league.zona_fantasma import Sandbox


class FakeModel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []  # copia de los mensajes que recibió en cada llamada

    def chat(self, messages, tools=None):
        self.seen.append(list(messages))
        return self.replies.pop(0)


def llamada(name, **args):
    return ModelReply("", [ToolCall(name, args)])


def hacer_agente(tmp_path, replies, approve=approve_all, max_steps=10):
    model = FakeModel(replies)
    agent = Agent(model, default_tools(Sandbox(tmp_path)), approve, "sistema", max_steps)
    return agent, model


def test_respuesta_directa_sin_herramientas(tmp_path):
    agent, _ = hacer_agente(tmp_path, [ModelReply("hola")])
    eventos = list(agent.run("hey"))
    assert [(e.kind, e.text) for e in eventos] == [("answer", "hola")]


def test_lee_un_archivo_y_responde(tmp_path):
    (tmp_path / "a.txt").write_text("contenido")
    agent, model = hacer_agente(tmp_path, [llamada("read_file", path="a.txt"), ModelReply("dice: contenido")])

    eventos = list(agent.run("lee a.txt"))

    assert [e.kind for e in eventos] == ["tool_call", "tool_result", "answer"]
    assert eventos[1].text == "contenido"
    # El resultado de la herramienta llegó al modelo en la segunda vuelta.
    assert model.seen[1][-1] == {"role": "tool", "tool_name": "read_file", "content": "contenido"}


def test_escritura_aprobada(tmp_path):
    agent, _ = hacer_agente(tmp_path, [llamada("write_file", path="n.txt", content="hola"), ModelReply("listo")])
    list(agent.run("crea n.txt"))
    assert (tmp_path / "n.txt").read_text() == "hola"


def test_escritura_denegada_no_toca_el_disco_y_el_modelo_lo_sabe(tmp_path):
    agent, model = hacer_agente(
        tmp_path, [llamada("write_file", path="n.txt", content="hola"), ModelReply("preparé n.txt")], approve=deny_all
    )
    eventos = list(agent.run("crea n.txt"))
    assert not (tmp_path / "n.txt").exists()
    assert eventos[-1].kind == "notice" and "descartados" in eventos[-1].text
    # En el turno siguiente el modelo recibe el aviso de que se rechazó.
    model.replies.append(ModelReply("ok"))
    list(agent.run("gracias"))
    assert "rechazó" in model.seen[-1][-1]["content"]


def test_el_approver_recibe_un_solo_diff_combinado(tmp_path):
    recibido = []

    def approver(name, args, preview):
        recibido.append((name, args, preview))
        return False

    (tmp_path / "a.txt").write_text("uno\n")
    agent, _ = hacer_agente(
        tmp_path,
        [
            llamada("write_file", path="a.txt", content="dos\n"),
            llamada("write_file", path="b.txt", content="nuevo\n"),
            ModelReply("listo"),
        ],
        approver,
    )
    list(agent.run("cambia"))
    assert len(recibido) == 1  # una sola aprobación, no una por edición
    name, args, preview = recibido[0]
    assert name == "aplicar_cambios" and args["archivos"] == ["a.txt", "b.txt"]
    assert "+dos" in preview and "+nuevo" in preview


def test_sin_cambios_preparados_no_se_pide_aprobacion(tmp_path):
    llamadas = []
    agent, _ = hacer_agente(tmp_path, [llamada("list_dir"), ModelReply("nada")], lambda *a: llamadas.append(a) or True)
    list(agent.run("x"))
    assert llamadas == []


def test_diez_ediciones_una_aprobacion_y_el_disco_cambia_al_final(tmp_path):
    (tmp_path / "a.txt").write_text("0\n")
    pasos = [llamada("edit_file", path="a.txt", old=str(i), new=str(i + 1)) for i in range(10)]
    aprobaciones = []
    agent, _ = hacer_agente(tmp_path, pasos + [ModelReply("hecho")], lambda *a: aprobaciones.append(a) or True, max_steps=12)
    list(agent.run("incrementa"))
    assert len(aprobaciones) == 1
    assert (tmp_path / "a.txt").read_text() == "10\n"


def test_herramienta_inexistente_se_devuelve_como_error_recuperable(tmp_path):
    agent, model = hacer_agente(tmp_path, [llamada("borrar_todo"), ModelReply("me equivoqué")])
    eventos = list(agent.run("x"))
    assert eventos[-1].kind == "answer"
    assert "no existe" in model.seen[1][-1]["content"]


def test_argumentos_invalidos_no_rompen_el_bucle(tmp_path):
    agent, model = hacer_agente(tmp_path, [llamada("read_file", ruta="a.txt"), ModelReply("ok")])
    list(agent.run("x"))
    assert "argumentos inválidos" in model.seen[1][-1]["content"]


def test_intento_de_escapar_del_sandbox_se_devuelve_como_error(tmp_path):
    agent, model = hacer_agente(tmp_path, [llamada("read_file", path="../../etc/passwd"), ModelReply("ok")])
    list(agent.run("x"))
    assert "SandboxError" in model.seen[1][-1]["content"]


def test_max_steps_corta_los_bucles(tmp_path):
    agent, _ = hacer_agente(tmp_path, [llamada("list_dir")] * 5, max_steps=3)
    eventos = list(agent.run("x"))
    assert eventos[-1].kind == "error"
    assert sum(e.kind == "tool_call" for e in eventos) == 3


def test_reset_conserva_solo_el_prompt_de_sistema(tmp_path):
    agent, _ = hacer_agente(tmp_path, [ModelReply("a")])
    list(agent.run("hola"))
    agent.reset()
    assert agent.messages == [{"role": "system", "content": "sistema"}]


def test_roles_por_defecto_cargan():
    from justice_league.config import DEFAULT_ROLES_PATH, load_settings

    settings = load_settings(DEFAULT_ROLES_PATH)
    assert settings.roles["superman"].enabled
    assert {"superman", "batman", "terrific", "wonder_woman", "robin"} <= set(settings.roles)
    assert sum(r.enabled for r in settings.roles.values()) == 1  # Fase 0: solo Superman


class FakeStreamModel:
    def stream(self, messages, tools=None):
        yield "ho"
        yield "la"
        yield ModelReply("hola")


def test_con_streaming_emite_tokens_y_luego_la_respuesta(tmp_path):
    agent = Agent(FakeStreamModel(), default_tools(Sandbox(tmp_path)), approve_all, "sistema")
    eventos = list(agent.run("hey"))
    assert [(e.kind, e.text) for e in eventos] == [("token", "ho"), ("token", "la"), ("answer", "hola")]
    assert agent.messages[-1] == {"role": "assistant", "content": "hola"}


def test_emite_stats_cuando_hay_metricas(tmp_path):
    agent, _ = hacer_agente(tmp_path, [ModelReply("hola", prompt_tokens=10, eval_tokens=5, eval_seconds=0.5)])
    eventos = list(agent.run("hey"))
    assert [e.kind for e in eventos] == ["stats", "answer", "summary"]
    assert eventos[0].args["prompt_tokens"] == 10 and eventos[0].args["eval_seconds"] == 0.5


def test_resumen_del_turno_suma_las_llamadas_y_mide_el_contexto_de_la_ultima(tmp_path):
    (tmp_path / "a.txt").write_text("hola")
    primera = ModelReply("", [ToolCall("read_file", {"path": "a.txt"})], prompt_tokens=100, eval_tokens=10,
                         eval_seconds=1.0, load_seconds=5.0, first_token_seconds=6.0)
    segunda = ModelReply("listo", prompt_tokens=150, eval_tokens=30, eval_seconds=1.0, load_seconds=0.1)
    agent, _ = hacer_agente(tmp_path, [primera, segunda])
    agent.model.num_ctx = 8192
    resumen = list(agent.run("lee a.txt"))[-1]
    assert resumen.kind == "summary"
    a = resumen.args
    assert a["calls"] == 2 and a["eval_tokens"] == 40 and a["eval_seconds"] == 2.0
    assert a["load_seconds"] == pytest.approx(5.1) and a["first_token_seconds"] == 6.0
    assert a["context_tokens"] == 180 and a["num_ctx"] == 8192


def test_el_tiempo_del_turno_no_cuenta_la_espera_de_aprobacion(tmp_path):
    import time

    pedidas = []

    def lento(name, args, preview):
        pedidas.append(name)
        time.sleep(0.3)
        return True

    reply = ModelReply("", [ToolCall("write_file", {"path": "n.txt", "content": "x"})], eval_tokens=1, eval_seconds=0.1)
    agent, _ = hacer_agente(tmp_path, [reply, ModelReply("hecho", eval_tokens=1, eval_seconds=0.1)], approve=lento)
    resumen = list(agent.run("crea n.txt"))[-1]
    assert pedidas and (tmp_path / "n.txt").read_text() == "x"
    assert resumen.kind == "summary" and resumen.args["seconds"] < 0.2


def test_llamada_repetida_que_fallo_se_bloquea_sin_ejecutarse(tmp_path):
    agent, model = hacer_agente(
        tmp_path, [llamada("read_file", path="no.txt"), llamada("read_file", path="no.txt"), ModelReply("me rindo")]
    )
    eventos = list(agent.run("lee no.txt"))
    assert eventos[-1].kind == "answer"
    assert model.seen[1][-1]["content"].startswith("ERROR")
    assert model.seen[2][-1]["content"].startswith("REPETIDO")


def test_si_insiste_se_corta_el_bucle(tmp_path):
    agent, _ = hacer_agente(tmp_path, [llamada("read_file", path="no.txt")] * 6)
    eventos = list(agent.run("lee no.txt"))
    assert eventos[-1].kind == "error" and "misma llamada" in eventos[-1].text
    assert sum(e.kind == "tool_call" for e in eventos) == 3  # 1 ejecutada + 2 bloqueadas


def test_repetir_una_llamada_exitosa_no_se_bloquea(tmp_path):
    agent, _ = hacer_agente(tmp_path, [llamada("list_dir"), llamada("list_dir"), llamada("list_dir"), ModelReply("ok")])
    eventos = list(agent.run("x"))
    assert eventos[-1].kind == "answer"


def test_el_bloqueo_se_reinicia_en_cada_turno(tmp_path):
    agent, model = hacer_agente(
        tmp_path, [llamada("read_file", path="no.txt"), ModelReply("a"), llamada("read_file", path="no.txt"), ModelReply("b")]
    )
    list(agent.run("uno"))
    list(agent.run("dos"))
    assert model.seen[3][-1]["content"].startswith("ERROR")  # se vuelve a ejecutar, no "REPETIDO"
