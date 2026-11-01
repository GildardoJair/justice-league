# Diario de experimentos

Una entrada por experimento. Regla: escribe la **hipótesis antes** de correrlo y la conclusión después. Si el resultado contradice la hipótesis, mejor: ahí se aprende.

## Plantilla

```
### E-XXX · título corto
- **Fecha:**
- **Hipótesis:**
- **Qué cambié (y qué dejé fijo):**
- **Cómo medí:**
- **Resultado:**
- **Conclusión / siguiente paso:**
```

## Cola de experimentos (por correr)

### E-001 · Contexto máximo vs. VRAM (qwen2.5:14b)
- **Hipótesis:** el 14B cuantizado deja poco margen en 12 GB; con `num_ctx` alto, Ollama pasará capas a CPU y la velocidad caerá.
- **Qué variar:** `num_ctx` (4k, 8k, 16k, 32k). Fijo: modelo, prompt, temperatura.
- **Cómo medir:** `ollama ps` (¿100 % GPU?), tokens/s de `ollama run --verbose`, `nvidia-smi` durante la generación.
- **Resultado:** _pendiente_

### E-002 · Fiabilidad de tool calling por modelo
- **Hipótesis:** los modelos grandes llaman bien a las herramientas y los chicos fallan más (argumentos mal formados, herramientas inventadas).
- **Qué variar:** modelo (`qwen2.5:14b`, `qwen2.5:7b-instruct`, `llama3.1:8b`, `llama3.2`, `mistral:7b`). Fijo: tarea y herramientas.
- **Cómo medir:** una tarea de 5 pasos (listar, leer, editar con diff, crear archivo, resumir), 10 corridas por modelo. Contar éxitos y tipo de fallo.
- **Resultado:** _pendiente_

### E-003 · Costo de cambiar de modelo
- **Hipótesis:** descargar un 14B y cargar un 7B cuesta varios segundos por delegación, lo que penaliza el multi-agente en 12 GB.
- **Cómo medir:** tiempo de la primera respuesta tras alternar Superman ↔ Mr. Terrific, con y sin `OLLAMA_KEEP_ALIVE` alto.
- **Resultado:** _pendiente_

## Bitácora

_(sin entradas todavía)_
