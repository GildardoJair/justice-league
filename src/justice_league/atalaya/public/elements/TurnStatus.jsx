// Indicador mientras se espera al modelo: "Cargando <modelo>…" o "Pensando…" con un contador.
// props: { phase: "loading" | "thinking", model, elapsed }
// `elapsed` son los segundos de espera cuando Python envió estas props; desde ahí cuenta el navegador.
import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";

export default function TurnStatus() {
  const [base, setBase] = useState(() => Date.now() - (props.elapsed || 0) * 1000);
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    setBase(Date.now() - (props.elapsed || 0) * 1000);
  }, [props.phase, props.elapsed]);

  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 200);
    return () => clearInterval(id);
  }, []);

  const seconds = Math.max(0, (now - base) / 1000);
  const label = props.phase === "loading" ? `Cargando ${props.model}…` : "Pensando…";
  return (
    <div className="flex items-center gap-2 text-sm text-muted-foreground">
      <Loader2 className="h-4 w-4 animate-spin" />
      <span>{label}</span>
      <span className="tabular-nums">{seconds.toFixed(1)} s</span>
    </div>
  );
}
