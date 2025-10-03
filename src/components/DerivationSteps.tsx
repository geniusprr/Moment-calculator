"use client";

import { BlockMath } from "react-katex";

interface DerivationStepsProps {
  steps: string[];
}

export function DerivationSteps({ steps }: DerivationStepsProps) {
  if (!steps.length) {
    return (
      <div className="panel-muted p-6 text-sm text-slate-400">
        Türetme adımlarını görmek için çözücüyü çalıştırın.
      </div>
    );
  }

  return (
    <div className="panel space-y-4 p-6">
      <div>
        <span className="tag">Öğretim Modu</span>
        <p className="text-sm text-slate-400">Denge denklemleri adım adım açıklanmıştır</p>
      </div>
      <ol className="space-y-4 text-slate-100">
        {steps.map((step, index) => (
          <li key={`${step}-${index}`} className="panel-muted border border-slate-800/60 p-4">
            <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-slate-400">
              Adım {index + 1}
            </p>
            <BlockMath math={step} />
          </li>
        ))}
      </ol>
    </div>
  );
}
