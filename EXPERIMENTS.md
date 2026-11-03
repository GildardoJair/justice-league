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
- **Resultado:** ver la entrada del 2026-11-03 en la bitácora. Hipótesis confirmada: el límite en 12 GB está entre 8k y 16k.

### E-002 · Fiabilidad de tool calling por modelo
- **Hipótesis:** los modelos grandes llaman bien a las herramientas y los chicos fallan más (argumentos mal formados, herramientas inventadas).
- **Qué variar:** modelo (`qwen2.5:14b`, `qwen2.5:7b-instruct`, `llama3.1:8b`, `llama3.2`, `mistral:7b`). Fijo: tarea y herramientas.
- **Cómo medir:** una tarea de 5 pasos (listar, leer, editar con diff, crear archivo, resumir), 10 corridas por modelo. Contar éxitos y tipo de fallo.
- **Resultado:** _pendiente_

### E-003 · Costo de cambiar de modelo
- **Hipótesis:** descargar un 14B y cargar un 7B cuesta varios segundos por delegación, lo que penaliza el multi-agente en 12 GB.
- **Cómo medir:** tiempo de la primera respuesta tras alternar Superman ↔ Mr. Terrific, con y sin `OLLAMA_KEEP_ALIVE` alto.
- **Resultado:** _pendiente_

### E-004 · Offload híbrido de un modelo denso grande
- **Hipótesis:** un modelo denso de ~30B (≈18-20 GB cuantizado) con más de la mitad de las capas en RAM irá a la velocidad de la CPU (3-6 tok/s), porque las capas se ejecutan en serie y cada token espera a la parte más lenta.
- **Qué variar:** número de capas en GPU (`num_gpu`). Fijo: modelo, prompt, `num_ctx` (8k), temperatura.
- **Cómo medir:** tok/s de la barra de estado de `jl chat`, porcentaje GPU/CPU en `ollama ps`, VRAM con `nvidia-smi`.
- **Resultado:** _pendiente_

### E-005 · MoE con reparto GPU/RAM frente a un denso
- **Hipótesis:** `qwen3:30b-a3b` (≈19 GB, ~3B parámetros activos por token) rendirá cerca de `qwen2.5:14b` en tok/s aunque no quepa entero en la VRAM, porque por token solo lee una fracción de sus pesos. `gpt-oss:20b` (≈13 GB) servirá de punto de comparación casi en GPU.
- **Qué variar:** modelo (`qwen2.5:14b` denso, `qwen3:30b-a3b`, `gpt-oss:20b`). Fijo: prompt, `num_ctx` (8k), temperatura.
- **Cómo medir:** tok/s, `ollama ps` (reparto GPU/CPU) y `nvidia-smi`. Anotar también si el tool calling funciona (enlaza con E-002).
- **Resultado:** _pendiente_

### E-006 · Modelo MoE enorme con pesos en el SSD (mmap)
- **Hipótesis:** `gpt-oss:120b` (≈65 GB, MoE con ~5B activos) no cabe en 12 GB de VRAM + 31 GB de RAM, pero con mmap funcionará a ~1 tok/s o menos, porque por token solo lee ~3 GB de expertos. Un denso equivalente (`llama3.3:70b` Q4, ≈42 GB) irá mucho peor (~0,3 tok/s).
- **Línea base medida (2026-11-03):** VRAM ≈ 360 GB/s (especificación de la RTX 3060, no medido); RAM ≈ 33 GB/s lectura de 1 hilo con numpy (cota inferior); SSD Kingston SNV3S ≈ 2,1-2,6 GB/s en lectura secuencial (`dd` con `iflag=direct`).
- **Qué variar:** modelo (`gpt-oss:120b`, `llama3.3:70b` Q4). Fijo: prompt, `num_ctx` (8k), temperatura.
- **Cómo medir:** tok/s de la barra de estado, tiempo hasta el primer token, `ollama ps`, lecturas del disco con `iostat` o `nvidia-smi`. Anotar si Ollama acepta cargar el modelo.
- **Aviso:** la descarga son ≈65 GB (≈3 h a 5,9 MB/s).
- **Resultado:** _pendiente_

## Bitácora

### 2026-11-03 · E-001 · Contexto máximo vs. VRAM (`qwen2.5:14b`)
- **Cómo lo corrí:** para cada `num_ctx` (4096, 8192, 16384, 32768) descargué el modelo (`ollama stop`) para forzar la recarga con ese contexto, y lancé 3 generaciones seguidas de 200 tokens (`num_predict`) por la API `/api/chat` en streaming. Mismo prompt ("Explica en español, en un párrafo, qué es un generador en Python y da un ejemplo corto."), temperatura 0,2, `think: false`. Los tok/s salen de `eval_count / eval_duration` que reporta Ollama. A mitad de la segunda generación leí `ollama ps` y `nvidia-smi`. Equipo: RTX 3060 12 GB, 31 GB de RAM, Ollama local. El script quedó en el scratchpad de la sesión, no en el repo.
- **Resultado:**

| `num_ctx` | Tamaño según `ollama ps` | Reparto CPU/GPU | tok/s (3 corridas) | VRAM usada |
|---|---|---|---|---|
| 4.096 | 9,7 GB | 0 % / 100 % | 34,4 · 34,3 · 34,2 | 9.716 MiB |
| 8.192 | 10 GB | 0 % / 100 % | 34,2 · 34,3 · 34,2 | 10.442 MiB |
| 16.384 | 13 GB | 15 % / 85 % | 15,5 · 15,5 · 15,5 | 10.606 MiB |
| 32.768 | 18 GB | 39 % / 61 % | 8,5 · 8,6 · 8,6 | 8.770 MiB |

- **Conclusión:**
  - Hipótesis confirmada. Hasta 8k todo cabe en la GPU y la velocidad es idéntica (~34 tok/s). A 16k Ollama pasa un 15 % del modelo a la CPU y la velocidad **cae a menos de la mitad** (15,5 tok/s). A 32k el 39 % va a la CPU y baja a ~8,5 tok/s, una cuarta parte.
  - El límite práctico con 12 GB está entre 8k y 16k. No medí valores intermedios (por ejemplo 12k).
  - El contexto se reserva entero al cargar: `ollama ps` muestra el tamaño total aunque la conversación use poco. Por eso `num_ctx` cuesta VRAM aunque no se llene.
  - A 32k la VRAM usada baja (8.770 MiB) porque se mandan más capas a la CPU para hacer sitio al cache.
  - Los tok/s son muy estables entre corridas (variación <1 %), así que con 3 repeticiones basta para este experimento.
  - Los tiempos de carga (2-4 s) no son comparables con los de otras pruebas: los pesos ya estaban en la caché de Linux.
  - **Siguiente paso:** probar valores intermedios (10k, 12k, 14k) para afinar el límite, y repetir con el KV cache cuantizado (`OLLAMA_KV_CACHE_TYPE=q8_0` con `OLLAMA_FLASH_ATTENTION=1`), que debería correr ese límite hacia arriba.
