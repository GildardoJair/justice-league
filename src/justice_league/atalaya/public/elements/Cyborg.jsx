// Cyborg: panel lateral con el estado del equipo (GPU, Ollama, CPU, RAM).
// props: { url, interval_ms }
// Consulta `url` cada `interval_ms` solo mientras el panel está abierto y la pestaña visible:
// al cerrar el panel el componente se desmonta y deja de preguntar.
import { useEffect, useState } from "react";

const HISTORY = 30; // puntos del mini-gráfico (30 × 2 s = 1 minuto)
const WARN = "#d97706";
const HOT = "#ef4444";

const gib = (n) => `${(n / 2 ** 30).toFixed(1)} GiB`;
const level = (value, warn, hot) => (value == null ? null : value >= hot ? HOT : value >= warn ? WARN : null);

function Section({ title, children }) {
  return (
    <div className="flex flex-col gap-2 border-b pb-2" style={{ paddingBottom: 12 }}>
      <div className="text-xs font-semibold uppercase text-muted-foreground" style={{ letterSpacing: "0.05em" }}>
        {title}
      </div>
      {children}
    </div>
  );
}

function Row({ label, value, color }) {
  return (
    <div className="flex justify-between gap-3 text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium tabular-nums truncate" style={color ? { color } : undefined}>
        {value}
      </span>
    </div>
  );
}

function Bar({ percent, warn = 80, hot = 95 }) {
  const p = Math.max(0, Math.min(100, percent || 0));
  return (
    <div className="w-full rounded-full overflow-hidden bg-muted" style={{ height: 6 }}>
      <div style={{ width: `${p}%`, height: "100%", background: level(p, warn, hot) || "hsl(var(--primary))", transition: "width .4s" }} />
    </div>
  );
}

function Spark({ values }) {
  if (values.length < 2) return null;
  const w = 100, h = 24;
  // Crece desde la derecha, como un monitor del sistema: el punto más nuevo siempre al borde.
  const n = values.length;
  const points = values.map((v, i) => `${w - ((n - 1 - i) / (HISTORY - 1)) * w},${h - (Math.min(100, v) / 100) * h}`).join(" ");
  return (
    <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" style={{ width: "100%", height: h }}>
      <polyline points={points} fill="none" stroke="hsl(var(--primary))" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function split(m) {
  if (!m.size) return "—";
  const gpu = Math.round((m.size_vram / m.size) * 100);
  return gpu >= 100 ? "100% GPU" : gpu <= 0 ? "100% CPU" : `${100 - gpu}% CPU / ${gpu}% GPU`;
}

function expiresIn(iso) {
  if (!iso) return null;
  const s = Math.round((new Date(iso).getTime() - Date.now()) / 1000);
  if (s <= 0 || s > 10 * 365 * 86400) return null; // ya vencido o "para siempre" (keep_alive -1)
  return s >= 60 ? `${Math.round(s / 60)} min` : `${s} s`;
}

export default function Cyborg() {
  const [snap, setSnap] = useState(null);
  const [error, setError] = useState(false);
  const [history, setHistory] = useState({ gpu: [], cpu: [] });

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      if (document.hidden) return; // pestaña en segundo plano: no gastar nada
      try {
        const res = await fetch(props.url, { cache: "no-store" });
        if (!res.ok) throw new Error(res.status);
        const data = await res.json();
        if (!alive) return;
        setSnap(data);
        setError(false);
        setHistory((h) => ({
          gpu: [...h.gpu, data.gpu ? data.gpu.util_percent : 0].slice(-HISTORY),
          cpu: [...h.cpu, data.cpu.percent].slice(-HISTORY),
        }));
      } catch (e) {
        if (alive) setError(true);
      }
    };
    tick();
    const id = setInterval(tick, props.interval_ms || 2000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [props.url, props.interval_ms]);

  if (!snap) {
    return <div className="text-sm text-muted-foreground">{error ? "Sin datos: ¿sigue corriendo Atalaya?" : "Leyendo sensores…"}</div>;
  }

  const { gpu, cpu, ram, ollama } = snap;
  const vram = gpu ? (gpu.mem_used / gpu.mem_total) * 100 : 0;
  const ramPct = (ram.used / ram.total) * 100;

  return (
    <div className="flex flex-col gap-3">
      <Section title="GPU">
        {gpu ? (
          <>
            <div className="text-sm font-medium truncate">{gpu.name}</div>
            <Row label="Uso" value={`${gpu.util_percent}%`} />
            <Bar percent={gpu.util_percent} warn={101} hot={101} />
            <Spark values={history.gpu} />
            <Row label="VRAM" value={`${gib(gpu.mem_used)} / ${gib(gpu.mem_total)}`} color={level(vram, 90, 97)} />
            <Bar percent={vram} warn={90} hot={97} />
            {gpu.temp_c != null && <Row label="Temperatura" value={`${gpu.temp_c} °C`} color={level(gpu.temp_c, 80, 87)} />}
            {gpu.power_w != null && <Row label="Consumo" value={`${Math.round(gpu.power_w)} / ${Math.round(gpu.power_limit_w)} W`} />}
          </>
        ) : (
          <div className="text-sm text-muted-foreground">No disponible (sin GPU NVIDIA o sin driver).</div>
        )}
      </Section>

      <Section title="Ollama">
        {ollama == null && <div className="text-sm" style={{ color: HOT }}>No responde.</div>}
        {ollama && ollama.length === 0 && <div className="text-sm text-muted-foreground">Ningún modelo cargado.</div>}
        {(ollama || []).map((m) => {
          const left = expiresIn(m.expires_at);
          return (
            <div key={m.name} className="flex flex-col gap-1">
              <div className="text-sm font-medium truncate">{m.name}</div>
              <Row label="Tamaño" value={gib(m.size)} />
              <Row label="Reparto" value={split(m)} color={m.size_vram < m.size ? WARN : null} />
              {m.context_length && <Row label="Contexto" value={m.context_length.toLocaleString("es")} />}
              {left && <Row label="Se descarga en" value={left} />}
            </div>
          );
        })}
      </Section>

      <Section title="CPU">
        <Row label="Uso" value={`${Math.round(cpu.percent)}% · ${cpu.cores} hilos`} />
        <Bar percent={cpu.percent} warn={101} hot={101} />
        <Spark values={history.cpu} />
        {cpu.temp_c != null && <Row label="Temperatura" value={`${Math.round(cpu.temp_c)} °C`} color={level(cpu.temp_c, 80, 90)} />}
      </Section>

      <Section title="RAM">
        <Row label="Usada" value={`${gib(ram.used)} / ${gib(ram.total)}`} color={level(ramPct, 85, 95)} />
        <Bar percent={ramPct} warn={85} hot={95} />
        {ram.swap_used > 2 ** 20 && <Row label="Swap" value={`${gib(ram.swap_used)} / ${gib(ram.swap_total)}`} color={WARN} />}
      </Section>

      <div className="text-xs text-muted-foreground">
        Se actualiza cada {(props.interval_ms || 2000) / 1000} s mientras el panel está abierto. Si lo cierras, escribe /cyborg para reabrirlo.
      </div>
    </div>
  );
}
