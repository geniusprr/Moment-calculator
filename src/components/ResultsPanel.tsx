import type { SupportReaction } from "@/types/beam";

interface ResultsPanelProps {
  reactions?: SupportReaction[];
  solveTimeMs?: number;
  warnings?: string[];
  error?: string | null;
}

const numberFormatter = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

export function ResultsPanel({ reactions, solveTimeMs, warnings = [], error }: ResultsPanelProps) {
  const hasReactions = reactions && reactions.length > 0;
  const totalReaction = hasReactions
    ? reactions.reduce((sum, reaction) => sum + (reaction.vertical ?? 0), 0)
    : 0;

  return (
    <div className="panel space-y-4 p-6">
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <span className="tag">Solution Summary</span>
          <p className="text-sm text-slate-400">Support reactions and solver metrics</p>
        </div>
        {solveTimeMs !== undefined && (
          <span className="rounded-full bg-slate-800/80 px-3 py-1 text-xs text-slate-300">
            {numberFormatter.format(solveTimeMs)} ms
          </span>
        )}
      </div>
      {error ? (
        <div className="panel-muted border border-rose-500/40 bg-rose-500/10 p-4 text-sm text-rose-200">
          {error}
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          {hasReactions ? (
            reactions.map((reaction) => (
              <div key={reaction.support_id} className="panel-muted p-4">
                <p className="text-xs uppercase tracking-wide text-slate-400">
                  R<sub>{reaction.support_id}</sub> ({reaction.support_type})
                </p>
                <p className="text-3xl font-semibold text-cyan-300">
                  {numberFormatter.format(reaction.vertical)} kN
                </p>
                <p className="text-xs text-slate-500">x = {numberFormatter.format(reaction.position)} m</p>
              </div>
            ))
          ) : (
            <div className="panel-muted col-span-full p-4 text-sm text-slate-400">
              Run the solver to see reactions.
            </div>
          )}
          {hasReactions && (
            <div className="panel-muted col-span-full p-4 text-sm text-slate-300">
              <p className="font-medium text-slate-200">Equilibrium check</p>
              <p>Sum of reactions = {numberFormatter.format(totalReaction)} kN</p>
            </div>
          )}
        </div>
      )}

      {warnings.length > 0 && (
        <div className="panel-muted border border-amber-400/40 bg-amber-500/10 p-4 text-xs text-amber-100">
          <p className="mb-2 font-semibold">Warnings</p>
          <ul className="space-y-2">
            {warnings.map((warning) => (
              <li key={warning}>- {warning}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}


