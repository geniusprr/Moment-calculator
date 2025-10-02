"use client";

import dynamic from "next/dynamic";
import { useMemo } from "react";
import type { Data, Layout } from "plotly.js";

interface BeamDiagramsProps {
  x: number[];
  shear: number[];
  moment: number[];
  normal: number[];
  loading?: boolean;
}

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

function diagramLayout(title: string, color: string): Partial<Layout> {
  return {
    paper_bgcolor: "rgba(15,23,42,0)",
    plot_bgcolor: "rgba(15,23,42,0)",
    margin: { l: 50, r: 20, t: 30, b: 40 },
    xaxis: {
      title: { text: "x (m)", font: { color: "#cbd5f5" } },
      tickfont: { color: "#94a3b8" },
      gridcolor: "rgba(148,163,184,0.2)",
      zerolinecolor: "rgba(148,163,184,0.35)"
    },
    yaxis: {
      title: { text: title, font: { color } },
      tickfont: { color: "#94a3b8" },
      gridcolor: "rgba(148,163,184,0.2)",
      zerolinecolor: "rgba(148,163,184,0.35)"
    },
    transition: { duration: 300, easing: "cubic-in-out" }
  };
}

function diagramData(x: number[], values: number[], color: string, label: string): Data[] {
  return [
    {
      x,
      y: values,
      type: "scatter",
      mode: "lines",
      line: { color, width: 3 },
      fill: "tozeroy",
      fillcolor: `${color}1A`,
      hovertemplate: `x = %{x:.2f} m<br>${label} = %{y:.2f}<extra></extra>`
    }
  ];
}

export function BeamDiagrams({ x, shear, moment, normal, loading = false }: BeamDiagramsProps) {
  const shearPlot = useMemo(
    () => ({ data: diagramData(x, shear, "#38bdf8", "V"), layout: diagramLayout("V (kN)", "#38bdf8") }),
    [x, shear]
  );
  const momentPlot = useMemo(
    // Flip moment sign so that bottom is negative and top is positive
    () => ({ data: diagramData(x, moment.map((v) => -v), "#a855f7", "M"), layout: diagramLayout("M (kNm)", "#a855f7") }),
    [x, moment]
  );
  const normalPlot = useMemo(
    () => ({ data: diagramData(x, normal, "#22d3ee", "N"), layout: diagramLayout("N (kN)", "#22d3ee") }),
    [x, normal]
  );

  const hasData = x.length > 0;

  return (
    <div className="panel space-y-4 p-6">
      <div className="flex items-center justify-between">
        <div>
          <span className="tag">Diagrams</span>
          <p className="text-sm text-slate-400">Shear, moment and axial force stacked for quick reading</p>
        </div>
        {loading && <span className="text-xs uppercase tracking-wide text-slate-400">Solving...</span>}
      </div>
      {hasData ? (
        <div className="space-y-6">
          {/* Put Normal force at the top */}
          <Plot data={normalPlot.data} layout={normalPlot.layout} config={{ displayModeBar: false, responsive: true }} style={{ width: "100%", height: "260px" }} />
          {/* Keep Shear next */}
          <Plot data={shearPlot.data} layout={shearPlot.layout} config={{ displayModeBar: false, responsive: true }} style={{ width: "100%", height: "260px" }} />
          {/* Moment at the bottom, with flipped sign */}
          <Plot data={momentPlot.data} layout={momentPlot.layout} config={{ displayModeBar: false, responsive: true }} style={{ width: "100%", height: "260px" }} />
        </div>
      ) : (
        <div className="panel-muted flex h-[360px] items-center justify-center text-sm text-slate-500">
          Run a solution to see the diagrams.
        </div>
      )}
    </div>
  );
}

