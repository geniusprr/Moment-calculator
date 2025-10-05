import type { AreaMethodVisualization, DetailedSolution } from "@/types/beam";

interface AreaMethodDiagramProps {
    diagram: NonNullable<DetailedSolution["diagram"]>;
    visualization: AreaMethodVisualization;
}

const WIDTH = 340;
const HEIGHT = 280;
const SECTION_PADDING = 36;
const CHART_HEIGHT = (HEIGHT - SECTION_PADDING) / 2;

const trendLabels: Record<AreaMethodVisualization["trend"], string> = {
    increase: "Moment artar",
    decrease: "Moment azalır",
    constant: "Moment sabit",
};

export function AreaMethodDiagram({ diagram, visualization }: AreaMethodDiagramProps) {
    const { region, moment_segment, shape, area_value, trend } = visualization;

    if (diagram.x.length === 0 || region.x.length === 0 || moment_segment.x.length === 0) {
        return null;
    }

    const xMin = Math.min(...diagram.x);
    const xMax = Math.max(...diagram.x);

    const shearMin = Math.min(...diagram.shear, 0);
    const shearMax = Math.max(...diagram.shear, 0);
    const momentMin = Math.min(...diagram.moment, 0);
    const momentMax = Math.max(...diagram.moment, 0);

    const shearZero = scaleValue(0, shearMin, shearMax);
    const momentZero = scaleValue(0, momentMin, momentMax);

    const shearLinePath = buildLinePath(diagram.x, diagram.shear, xMin, xMax, shearMin, shearMax, true);
    const momentLinePath = buildLinePath(diagram.x, diagram.moment, xMin, xMax, momentMin, momentMax, false);

    const highlightAreaPath = buildAreaPath(region.x, region.shear, xMin, xMax, shearMin, shearMax, shearZero);
    const highlightShearLinePath = buildLinePath(region.x, region.shear, xMin, xMax, shearMin, shearMax, true);
    const highlightMomentLinePath = buildLinePath(
        moment_segment.x,
        moment_segment.moment,
        xMin,
        xMax,
        momentMin,
        momentMax,
        false,
    );

    return (
        <div className="flex w-full max-w-sm flex-col gap-3">
            <div className="flex items-center justify-between">
                <div>
                    <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">Alan Yöntemi Vurgusu</p>
                    <p className="text-xs text-slate-500">{trendLabels[trend]} – {shape}</p>
                </div>
                <div className="rounded-full bg-cyan-500/10 px-3 py-1 text-xs font-medium text-cyan-300">
                    ΔM = {area_value.toFixed(2)} kN·m
                </div>
            </div>

            <div className="rounded-xl border border-cyan-500/30 bg-slate-950/60 p-4 shadow-inner">
                <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="h-72 w-full">
                    {/* Shear Chart */}
                    <g>
                        <rect
                            x={24}
                            y={16}
                            width={WIDTH - 48}
                            height={CHART_HEIGHT - 16}
                            rx={12}
                            fill="#0f172a"
                            fillOpacity={0.35}
                        />
                        {/* Axes */}
                        <line x1={scaleX(xMin, xMin, xMax)} y1={shearZero} x2={scaleX(xMax, xMin, xMax)} y2={shearZero} stroke="#475569" strokeDasharray="4 2" />
                        <line x1={scaleX(xMin, xMin, xMax)} y1={chartTop()} x2={scaleX(xMin, xMin, xMax)} y2={chartBottom()} stroke="#475569" strokeDasharray="2 3" />

                        {/* Total shear line */}
                        <path d={shearLinePath} fill="none" stroke="#334155" strokeWidth={1.5} />

                        {/* Highlight area */}
                        {highlightAreaPath && (
                            <path d={highlightAreaPath} fill="#22d3ee" fillOpacity={0.2} stroke="none" />
                        )}
                        <path d={highlightShearLinePath} fill="none" stroke="#22d3ee" strokeWidth={2.2} />

                        {/* Labels */}
                        <text x={scaleX((xMin + xMax) / 2, xMin, xMax)} y={chartTop() + 14} textAnchor="middle" fill="#94a3b8" fontSize={11}>
                            Kesme grafiği V(x)
                        </text>
                        <text x={scaleX(xMax, xMin, xMax)} y={shearZero - 6} textAnchor="end" fill="#64748b" fontSize={9}>
                            x (m)
                        </text>
                    </g>

                    {/* Moment Chart */}
                    <g transform={`translate(0, ${CHART_HEIGHT + 12})`}>
                        <rect
                            x={24}
                            y={4}
                            width={WIDTH - 48}
                            height={CHART_HEIGHT - 16}
                            rx={12}
                            fill="#0f172a"
                            fillOpacity={0.35}
                        />
                        <line x1={scaleX(xMin, xMin, xMax)} y1={momentZero} x2={scaleX(xMax, xMin, xMax)} y2={momentZero} stroke="#475569" strokeDasharray="4 2" />
                        <line x1={scaleX(xMin, xMin, xMax)} y1={chartBottom()} x2={scaleX(xMin, xMin, xMax)} y2={chartTop()} stroke="#475569" strokeDasharray="2 3" />

                        {/* Total moment line */}
                        <path d={momentLinePath} fill="none" stroke="#334155" strokeWidth={1.5} />

                        {/* Highlighted moment segment */}
                        <path d={highlightMomentLinePath} fill="none" stroke="#facc15" strokeWidth={2.4} />

                        {moment_segment.x.length > 0 && (
                            <circle
                                cx={scaleX(moment_segment.x[moment_segment.x.length - 1], xMin, xMax)}
                                cy={scaleMoment(moment_segment.moment[moment_segment.moment.length - 1], momentMin, momentMax)}
                                r={4}
                                fill="#facc15"
                            />
                        )}

                        <text x={scaleX((xMin + xMax) / 2, xMin, xMax)} y={chartTop() + 20} textAnchor="middle" fill="#f1f5f9" fontSize={11}>
                            Moment grafiği M(x)
                        </text>
                        <text x={scaleX(xMax, xMin, xMax)} y={momentZero - 6} textAnchor="end" fill="#64748b" fontSize={9}>
                            x (m)
                        </text>
                    </g>
                </svg>
            </div>
        </div>
    );
}

function scaleX(value: number, min: number, max: number): number {
    if (max - min === 0) return WIDTH / 2;
    const padding = 40;
    return padding + ((value - min) / (max - min)) * (WIDTH - padding * 2);
}

function chartTop(): number {
    return 24;
}

function chartBottom(): number {
    return CHART_HEIGHT - 8;
}

function scaleValue(value: number, min: number, max: number): number {
    const top = chartTop();
    const bottom = chartBottom();
    if (max - min === 0) {
        return (top + bottom) / 2;
    }
    const normalized = (value - min) / (max - min);
    return bottom - normalized * (bottom - top);
}

function scaleShear(value: number, min: number, max: number): number {
    return scaleValue(value, min, max);
}

function scaleMoment(value: number, min: number, max: number): number {
    return scaleValue(value, min, max);
}

function buildLinePath(
    xValues: number[],
    yValues: number[],
    xMin: number,
    xMax: number,
    yMin: number,
    yMax: number,
    isShear: boolean,
): string {
    if (xValues.length === 0) return "";

    const commands: string[] = [];
    for (let i = 0; i < xValues.length; i++) {
        const x = scaleX(xValues[i], xMin, xMax);
        const y = isShear ? scaleShear(yValues[i], yMin, yMax) : scaleMoment(yValues[i], yMin, yMax);
        commands.push(`${i === 0 ? "M" : "L"} ${x} ${y}`);
    }
    return commands.join(" ");
}

function buildAreaPath(
    xValues: number[],
    yValues: number[],
    xMin: number,
    xMax: number,
    yMin: number,
    yMax: number,
    zeroY: number,
): string | null {
    if (xValues.length === 0) return null;

    const points: { x: number; y: number }[] = [];
    for (let i = 0; i < xValues.length; i++) {
        points.push({
            x: scaleX(xValues[i], xMin, xMax),
            y: scaleShear(yValues[i], yMin, yMax),
        });
    }

    let path = `M ${points[0].x} ${zeroY}`;
    points.forEach((pt) => {
        path += ` L ${pt.x} ${pt.y}`;
    });
    path += ` L ${points[points.length - 1].x} ${zeroY} Z`;
    return path;
}
