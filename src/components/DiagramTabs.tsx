"use client";

import dynamic from "next/dynamic";
import { useMemo, useState } from "react";
import type { Data, Layout } from "plotly.js";

interface DiagramTabsProps {
  x: number[];
  shear: number[];
  moment: number[];
  loading?: boolean;
}

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const tabs = [
  { id: "shear", label: "Shear V(x)", color: "#38bdf8" },
  { id: "moment", label: "Moment M(x)", color: "#a855f7" },
] as const;

export function DiagramTabs({ x, shear, moment, loading = false }: DiagramTabsProps) {
  const [active, setActive] = useState<(typeof tabs)[number]["id"]>("shear");

  const { data, layout } = useMemo(() => {
    const isShear = active === "shear";
    const values = isShear ? shear : moment;
    const color = isShear ? "#38bdf8" : "#a855f7";
    const yTitle = isShear ? "V (kN)" : "M (kNm)";

    const chartData: Data[] = [
      {
        x,
        y: values,
        type: "scatter",
        mode: "lines",
        line: { color, width: 3 },
        fill: "tozeroy",
        fillcolor: `${color}1A`,
        hovertemplate: `x = %{x:.2f} m<br>${yTitle} = %{y:.2f}<extra></extra>`,
      },
    ];

    const chartLayout: Partial<Layout> = {
      paper_bgcolor: "rgba(15,23,42,0)",
      plot_bgcolor: "rgba(15,23,42,0)",
      margin: { l: 40, r: 20, t: 30, b: 40 },
      xaxis: {
        title: { text: "x (m)", font: { color: "#cbd5f5" } },
        tickfont: { color: "#94a3b8" },
        gridcolor: "rgba(148,163,184,0.18)",
        zerolinecolor: "rgba(148,163,184,0.35)",
      },
      yaxis: {
        title: { text: yTitle, font: { color: "#cbd5f5" } },
        tickfont: { color: "#94a3b8" },
        gridcolor: "rgba(148,163,184,0.18)",
        zerolinecolor: "rgba(148,163,184,0.35)",
      },
      transition: { duration: 300, easing: "cubic-in-out" },
    };

    return { data: chartData, layout: chartLayout };
  }, [active, moment, shear, x]);

  const hasData = x.length > 0;

  return (
    <div className="panel p-6">
      <div className="flex flex-wrap items-center justify-between gap-3 pb-4">
        <div className="flex flex-wrap gap-2">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActive(tab.id)}
              className={`rounded-full px-4 py-1.5 text-sm transition ${
                active === tab.id
                  ? "bg-slate-800 text-white shadow-inner"
                  : "bg-slate-900/40 text-slate-400 hover:text-white hover:bg-slate-800/70"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
        {loading ? (
          <span className="text-xs uppercase tracking-wide text-slate-400">Solving...</span>
        ) : (
          <span className="text-xs text-slate-500">Hover the curve to inspect values.</span>
        )}
      </div>
      {hasData ? (
        <Plot
          data={data}
          layout={layout}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: "100%", height: "320px" }}
        />
      ) : (
        <div className="panel-muted flex h-[320px] items-center justify-center text-sm text-slate-500">
          No data yet. Run a solve to plot diagrams.
        </div>
      )}
    </div>
  );
}
