// Métricas de un turno, como chips debajo de la respuesta.
// props: { parts: [{ kind, text }], level: "ok" | "warn" | "full" }
// Solo se usan clases que ya trae el CSS de Chainlit; los colores de aviso van en línea.
import { Badge } from "@/components/ui/badge";

const CONTEXT_COLOR = { warn: "#d97706", full: "#ef4444" };

export default function TurnMetrics() {
  const warn = CONTEXT_COLOR[props.level];
  return (
    <div className="flex flex-wrap items-center gap-1 mt-1">
      {props.parts.map((part) => {
        const alert = part.kind === "context" && warn;
        return (
          <Badge
            key={part.kind}
            variant="outline"
            className="text-xs font-normal tabular-nums text-muted-foreground"
            style={alert ? { color: warn, borderColor: warn } : undefined}
          >
            {part.text}
          </Badge>
        );
      })}
    </div>
  );
}
