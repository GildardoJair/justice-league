# Justice League

Un equipo local de agentes de IA para programación y ciencia de datos. Corre sobre modelos de [Ollama](https://ollama.com), sin enviar nada a la nube, y vive en dos caras que comparten el mismo cerebro:

- **`jl`**: la terminal.
- **Atalaya**: el chat web (Chainlit).

Es un proyecto de hobby para aprender: agentes, RAG, grafos de conocimiento y cómo se comportan los modelos locales en una RTX 3060 de 12 GB. Cada decisión de diseño deja una nota de qué concepto ilustra (búsquela en los docstrings) y cada experimento se anota en [`EXPERIMENTS.md`](EXPERIMENTS.md).

## Arquitectura

```
   Atalaya (Chainlit)        jl (Typer + Rich)        ← adaptadores delgados
            └──────────────┬──────────────┘
                     factory.build_agent()
                           │
                    Agent (bucle) ── eventos ──▶ quien consuma
              ┌────────┬───┴────┬──────────┐
           llm.py   tools.py  permissions  roles.toml
          (Ollama)     │       (guardián)
                  Zona Fantasma
                   (sandbox)
```

El núcleo no sabe nada de las interfaces: `Agent.run()` emite eventos y cada cara los muestra a su manera.

## El equipo

Un rol es **configuración** (modelo + prompt), no código: viven en [`roles.toml`](src/justice_league/roles.toml). Las piezas de infraestructura sí tienen su módulo.

| Pieza | Nombre | Qué es | Estado |
|---|---|---|---|
| Orquestador | **Superman** | `qwen2.5:14b`, planea y decide | activo (Fase 0) |
| Analista | **Batman** | coder 7B, analiza datos | declarado, Fase 4 |
| Programador | **Mr. Terrific** | coder 7B, escribe código | declarado |
| Gestor de contexto | **Wonder Woman** | 7B, resume sin distorsionar | declarado |
| Modelo rápido | **Robin** | `llama3.2`, enruta y titula | declarado |
| Recuperador | **Flash** | búsqueda vectorial + BM25 + rerank | `flash/`, Fase 1-2 |
| Almacén vectorial | **Fortaleza de la Soledad** | Chroma o LanceDB | `fortaleza/`, Fase 1 |
| Grafo de conocimiento | **Brainiac** | Kuzu o Neo4j | `brainiac/`, Fase 3 |
| Sandbox | **Zona Fantasma** | aislamiento de archivos y, luego, de código | `zona_fantasma/`, archivos listo |
| Memoria | *(sin nombre)* | recuerdo entre sesiones | pendiente |
| Guardián de permisos | *(sin nombre)* | aprobación humana | `permissions.py` |

## Puesta en marcha

Requiere [uv](https://docs.astral.sh/uv/) y Ollama corriendo.

```bash
uv sync                      # núcleo + CLI
uv sync --extra atalaya      # además, la interfaz web

uv run jl doctor             # ¿responde Ollama? ¿están los modelos?
uv run jl roles              # roles configurados
uv run jl chat -w ~/mi-proyecto
uv run jl chat -m qwen2.5:7b-instruct   # modelo más ligero para iterar
uv run jl atalaya -w ~/mi-proyecto
```

Para usar `jl` sin `uv run`: `uv tool install --editable .`

Pruebas (no necesitan Ollama ni GPU): `uv run pytest`

## Seguridad

El agente escribe archivos, así que desde el día uno:

- **Sandbox de rutas:** solo ve y toca la carpeta de trabajo (bloquea `../` y symlinks que escapen). `.git` es de solo lectura.
- **Aprobación humana:** toda escritura (`write_file`, `edit_file`) muestra un diff y espera tu sí.
- **Límite de pasos:** el bucle se corta tras 10 vueltas.

Límite conocido: esto valida rutas, no aísla procesos. Cuando llegue la ejecución de código (Fase 4) habrá que aislarla de verdad (contenedor o usuario sin privilegios).

## Hoja de ruta

- [x] **Fase 0, esqueleto:** núcleo, herramientas de archivos, guardián, CLI y Atalaya.
  - [x] Respuestas en *streaming* (CLI y Atalaya; falta probarlo con Ollama real).
  - [ ] Medir contexto máximo del 14B en 12 GB de VRAM (ver EXPERIMENTS.md).
- [ ] **Fase 1, RAG básico:** ingesta → chunking → embeddings → búsqueda → respuesta con citas.
- [ ] **Fase 2, RAG mejorado:** búsqueda híbrida, reranking y evaluación.
- [ ] **Fase 3, grafo:** extracción de entidades y relaciones, Cypher, GraphRAG.
- [ ] **Fase 4, agencia:** más roles, delegación entre agentes, ejecución de código aislada, MCP, memoria.
- [ ] **Fase 5, escalar:** modelos mayores, cuantización, observabilidad, quizá fine-tuning (LoRA).

### Temas por tratar

- Herramienta de estado del hardware: hoy el agente no puede consultar GPU/VRAM, CPU, RAM ni disco (p. ej. envolver `nvidia-smi`/`ollama ps` en una herramienta de solo lectura). Útil para E-001 y para que el agente sepa cuánta VRAM libre tiene antes de cambiar de modelo.

### Líneas de investigación a seguir

Un terreno donde se quiere aportar: hacer que los LLM (densos y MoE) rindan más con equipo modesto. Nombres y cifras anotados de memoria; verificar antes de citarlos.

- **Memoria y ancho de banda:** cuantización agresiva (AWQ, GPTQ, K-quants, BitNet), KV cache más pequeño (cuantizado, GQA, MLA, descarte de tokens), offloading con *prefetch* de expertos en MoE.
- **Cálculo por token:** decodificación especulativa (Medusa, EAGLE), atención eficiente (FlashAttention, Mamba/RWKV e híbridos), dispersión y salida temprana.
- **Razonamiento sin más tamaño:** cómputo en inferencia (pensar más, votar, verificar), aprendizaje por refuerzo con recompensas verificables, destilación, mejores datos y uso de herramientas.
- **Entrenar y adaptar con poco:** LoRA/QLoRA, fusión de modelos y Branch-Train-MiX, entrenamiento distribuido.

Para probar a escala doméstica: decodificación especulativa (7B de borrador + 14B), KV cuantizado (E-001), MoE con reparto GPU/RAM/SSD (E-005, E-006) y el bucle de ejecución y verificación del agente (Fase 4).

Los nombres de personajes de DC son un homenaje personal y se usan como nombres internos.
