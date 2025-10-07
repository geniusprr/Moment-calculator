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
  shearMarkers?: number[];
}

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

function diagramLayout(title: string, color: string, autorange: "reversed" | true = true): Partial<Layout> {
  return {
    paper_bgcolor: "rgba(15,23,42,0)",
    plot_bgcolor: "rgba(15,23,42,0)",
    margin: { l: 50, r: 20, t: 30, b: 40 },
    showlegend: false,
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
      zerolinecolor: "rgba(148,163,184,0.35)",
      autorange
    },
    transition: { duration: 300, easing: "cubic-in-out" }
  };
}

interface SignedSegment {
  x: number[];
  y: number[];
  sign: -1 | 0 | 1;
  midpoint: number;
  amplitude: number;
}

function determineSign(values: number[]): -1 | 0 | 1 {
  const EPS = 1e-9;
  for (const value of values) {
    if (Math.abs(value) > EPS) {
      return value > 0 ? 1 : -1;
    }
  }
  return 0;
}

function createSignedSegments(x: number[], values: number[]): SignedSegment[] {
  if (x.length < 2) {
    return [];
  }

  const EPS = 1e-9;
  const segments: SignedSegment[] = [];

  const pushSegment = (xs: number[], ys: number[]) => {
    const sign = determineSign(ys);
    if (sign === 0) {
      return;
    }
    const amplitude = Math.max(...ys.map((value) => Math.abs(value)));
    const start = xs[0];
    const end = xs[xs.length - 1];
    segments.push({
      x: [...xs],
      y: [...ys],
      sign,
      midpoint: start + (end - start) / 2,
      amplitude
    });
  };

  let segmentX: number[] = [x[0]];
  let segmentY: number[] = [values[0]];
  let segmentSign: -1 | 0 | 1 = Math.abs(values[0]) <= EPS ? 0 : values[0] > 0 ? 1 : -1;

  for (let i = 1; i < x.length; i += 1) {
    const prevX = x[i - 1];
    const prevY = values[i - 1];
    const currX = x[i];
    const currY = values[i];

    const currSignRaw: -1 | 0 | 1 = Math.abs(currY) <= EPS ? 0 : currY > 0 ? 1 : -1;

    const signChange = segmentSign !== 0 && currSignRaw !== 0 && currSignRaw !== segmentSign;

    if (signChange) {
      const lastIndex = segmentY.length - 1;
      let zeroPointX = segmentX[lastIndex];

      if (Math.abs(segmentY[lastIndex]) > EPS) {
        const zeroX = prevX + (currX - prevX) * (-prevY) / (currY - prevY);
        if (Number.isFinite(zeroX)) {
          zeroPointX = zeroX;
          segmentX.push(zeroX);
          segmentY.push(0);
        }
      }

      pushSegment(segmentX, segmentY);

      segmentX = [zeroPointX, currX];
      segmentY = [0, currY];
      segmentSign = currSignRaw;
      continue;
    }

    segmentX.push(currX);
    segmentY.push(currY);

    if (segmentSign === 0 && currSignRaw !== 0) {
      segmentSign = currSignRaw;
    }
  }

  pushSegment(segmentX, segmentY);

  return segments;
}

type DiagramOptions = {
  includeZeroBaseline?: boolean;
  segmentBySign?: boolean;
  fillToZero?: boolean;
  fillColor?: string;
};

function diagramData(
  x: number[],
  values: number[],
  color: string,
  label: string,
  options: DiagramOptions = {}
): Data[] {
  const { includeZeroBaseline = false, segmentBySign = false, fillToZero = false, fillColor } = options;

  const baseTrace: Data = {
    x,
    y: values,
    type: "scatter",
    mode: "lines",
    line: { color, width: 3 },
    hovertemplate: `x = %{x:.2f} m<br>${label} = %{y:.2f}<extra></extra>`,
    showlegend: false,
    ...(fillToZero && x.length >= 2
      ? {
        fill: "tozeroy" as const,
        fillcolor: fillColor ?? `${color}1A`
      }
      : {})
  };

  const traces: Data[] = [];

  if (segmentBySign && x.length >= 2) {
    const segments = createSignedSegments(x, values);

    segments.forEach((segment, index) => {
      const positive = segment.sign > 0;
      const fillColor = positive ? "rgba(34,197,94,0.22)" : "rgba(248,113,113,0.22)";
      const borderColor = positive ? "rgba(34,197,94,0.8)" : "rgba(248,113,113,0.8)";

      traces.push({
        x: segment.x,
        y: segment.y,
        type: "scatter",
        mode: "lines",
        line: { width: 1.5, color: borderColor },
        fill: "tozeroy",
        fillcolor: fillColor,
        hoverinfo: "skip",
        showlegend: false,
        name: positive ? `+ Bölge ${index + 1}` : `- Bölge ${index + 1}`
      });
    });
  }

  if (includeZeroBaseline && x.length >= 2) {
    traces.push({
      x: [x[0], x[x.length - 1]],
      y: [0, 0],
      type: "scatter",
      mode: "lines",
      line: { color: `${color}80`, width: 2, dash: "dot" },
      hoverinfo: "skip",
      showlegend: false
    });
  }

  traces.push(baseTrace);

  return traces;
}

export function BeamDiagrams({ x, shear, moment, normal, loading = false, shearMarkers = [] }: BeamDiagramsProps) {
  const shearPlot = useMemo(
    () => {
      const data = diagramData(x, shear, "#38bdf8", "T", { includeZeroBaseline: true, segmentBySign: true });

      if (shearMarkers.length > 0 && shear.length > 0) {
        const shearMin = Math.min(0, ...shear);
        const shearMax = Math.max(0, ...shear);
        const range = Math.max(1e-6, shearMax - shearMin);
        const padding = range * 0.08;
        const lineBottom = shearMin - padding;
        const lineTop = shearMax + padding;

        shearMarkers
          .filter((position) => position >= x[0] - 1e-6 && position <= x[x.length - 1] + 1e-6)
          .forEach((position) => {
            data.push({
              x: [position, position],
              y: [lineBottom, lineTop],
              type: "scatter",
              mode: "lines",
              line: { color: "rgba(148,163,184,0.5)", width: 1.5, dash: "dot" },
              hovertemplate: `x = ${position.toFixed(2)} m<extra>Bölge sınırı</extra>`,
              showlegend: false
            });
          });
      }

      return {
        data,
        layout: diagramLayout("T (kN)", "#38bdf8")
      };
    },
    [x, shear, shearMarkers]
  );
  const momentPlot = useMemo(
    // Reverse y-axis only: positive values show downward (below axis), negative values show upward (above axis)
    () => ({
      data: diagramData(x, moment, "#38bdf8", "M", {
        includeZeroBaseline: true,
        fillToZero: true,
        fillColor: "rgba(56,189,248,0.25)"
      }),
      layout: diagramLayout("M (kNm)", "#38bdf8", "reversed")
    }),
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
          <span className="tag">Diyagramlar</span>
          <p className="text-sm text-slate-400">Kesme, moment ve eksenel kuvvet diyagramları</p>
        </div>
        {loading && <span className="text-xs uppercase tracking-wide text-slate-400">Çözülüyor...</span>}
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
          Diyagramları görmek için çözüm çalıştırın.
        </div>
      )}
    </div>
  );
}

