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
        tmp_path, [llamada("write_file", path="n.txt", content="hola"), ModelReply("ok, no lo hago")], approve=deny_all
    )
    list(agent.run("crea n.txt"))
    assert not (tmp_path / "n.txt").exists()
    assert model.seen[1][-1]["content"].startswith("DENEGADO")


def test_el_approver_recibe_el_diff(tmp_path):
    recibido = {}

    def approver(name, args, preview):
        recibido.update(name=name, preview=preview)
        return False

    (tmp_path / "a.txt").write_text("uno\n")
    agent, _ = hacer_agente(tmp_path, [llamada("write_file", path="a.txt", content="dos\n"), ModelReply("ok")], approver)
    list(agent.run("cambia"))
    assert recibido["name"] == "write_file" and "+dos" in recibido["preview"]


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
    assert [e.kind for e in eventos] == ["stats", "answer"]
    assert eventos[0].args == {"prompt_tokens": 10, "eval_tokens": 5, "eval_seconds": 0.5}
