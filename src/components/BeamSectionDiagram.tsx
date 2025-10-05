import type { BeamContext, BeamSectionHighlight } from "@/types/beam";

interface BeamSectionDiagramProps {
    context: BeamContext;
    highlight: BeamSectionHighlight;
}

export function BeamSectionDiagram({ context, highlight }: BeamSectionDiagramProps) {
    const width = 340;
    const height = 160;
    const margin = 32;
    const beamY = height / 2;
    const effectiveLength = Math.max(context.length, 1e-6);

    const scaleX = (value: number) => {
        const clamped = Math.min(Math.max(value, 0), effectiveLength);
        return margin + (clamped / effectiveLength) * (width - 2 * margin);
    };

    const rawStart = Math.min(highlight.start, highlight.end);
    const rawEnd = Math.max(highlight.start, highlight.end);
    const highlightStart = scaleX(rawStart);
    const highlightEnd = scaleX(rawEnd);
    const isPointHighlight = Math.abs(highlight.end - highlight.start) < 1e-6;
    const bandWidth = Math.max(highlightEnd - highlightStart, 6);
    const bandX = isPointHighlight ? highlightStart - bandWidth / 2 : highlightStart;

    return (
        <div className="flex w-full max-w-sm flex-col gap-3">
            <div className="flex items-center justify-between">
                <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">Kesilen Bölge</p>
                {highlight.label && <span className="text-xs text-cyan-300">{highlight.label}</span>}
            </div>
            <div className="rounded-xl border border-cyan-500/30 bg-slate-950/60 p-4 shadow-inner">
                <svg viewBox={`0 0 ${width} ${height}`} className="h-36 w-full">
                    {/* Beam baseline */}
                    <line
                        x1={margin}
                        y1={beamY}
                        x2={width - margin}
                        y2={beamY}
                        stroke="#0f172a"
                        strokeWidth={12}
                        strokeLinecap="round"
                    />

                    {/* Supports */}
                    {context.supports.map((support) => {
                        const x = scaleX(support.position);
                        const baseY = beamY + 24;
                        const apexY = beamY + (support.type === "roller" ? 8 : 4);
                        return (
                            <g key={support.id}>
                                <polygon
                                    points={`${x - 12},${baseY} ${x + 12},${baseY} ${x},${apexY}`}
                                    fill="#38bdf8"
                                    opacity={0.7}
                                />
                                {support.type === "roller" && (
                                    <g>
                                        <circle cx={x - 6} cy={baseY + 9} r={4} fill="#94a3b8" />
                                        <circle cx={x + 6} cy={baseY + 9} r={4} fill="#94a3b8" />
                                    </g>
                                )}
                                <text
                                    x={x}
                                    y={baseY + 24}
                                    textAnchor="middle"
                                    fill="#e2e8f0"
                                    fontSize={10}
                                >
                                    {support.id}
                                </text>
                            </g>
                        );
                    })}

                    {/* Highlight */}
                    <rect
                        x={bandX}
                        y={beamY - 22}
                        width={bandWidth}
                        height={44}
                        rx={10}
                        fill="#22d3ee"
                        fillOpacity={0.18}
                        stroke="#22d3ee"
                        strokeOpacity={0.9}
                        strokeWidth={2}
                    />

                    {/* Highlight boundaries */}
                    <line
                        x1={highlightStart}
                        y1={beamY - 28}
                        x2={highlightStart}
                        y2={beamY + 28}
                        stroke="#22d3ee"
                        strokeDasharray="4 2"
                        strokeWidth={1.5}
                    />
                    {!isPointHighlight && (
                        <line
                            x1={highlightEnd}
                            y1={beamY - 28}
                            x2={highlightEnd}
                            y2={beamY + 28}
                            stroke="#22d3ee"
                            strokeDasharray="4 2"
                            strokeWidth={1.5}
                        />
                    )}

                    {/* Point marker */}
                    {isPointHighlight && (
                        <circle
                            cx={highlightStart}
                            cy={beamY}
                            r={8}
                            fill="#22d3ee"
                            fillOpacity={0.8}
                        />
                    )}

                    {/* Loads: point loads */}
                    {context.point_loads.map((load) => {
                        const x = scaleX(load.position);
                        const isDown = load.angle_deg <= -90 + 1e-3;
                        const arrowTipY = isDown ? beamY - 20 : beamY - 60;
                        const arrowTailY = isDown ? arrowTipY - 24 : arrowTipY + 24;
                        return (
                            <g key={load.id}>
                                <line
                                    x1={x}
                                    y1={arrowTailY}
                                    x2={x}
                                    y2={arrowTipY}
                                    stroke="#f97316"
                                    strokeWidth={2}
                                    strokeLinecap="round"
                                />
                                <polygon
                                    points={`${x - 5},${arrowTipY} ${x + 5},${arrowTipY} ${x},${arrowTipY + (isDown ? 10 : -10)}`}
                                    fill="#f97316"
                                />
                                <text x={x} y={arrowTailY - 8} textAnchor="middle" fill="#fda4af" fontSize={9}>
                                    {load.id}
                                </text>
                            </g>
                        );
                    })}

                    {/* Distributed loads */}
                    {context.udls.map((udl) => {
                        const x1 = scaleX(udl.start);
                        const x2 = scaleX(udl.end);
                        const arrowCount = Math.max(Math.floor((x2 - x1) / 28), 2);
                        const spacing = (x2 - x1) / (arrowCount - 1);
                        return (
                            <g key={udl.id}>
                                <rect
                                    x={x1}
                                    y={beamY - 70}
                                    width={Math.max(x2 - x1, 8)}
                                    height={6}
                                    fill="#a855f7"
                                    opacity={0.5}
                                />
                                {Array.from({ length: arrowCount }).map((_, index) => {
                                    const arrowX = x1 + index * spacing;
                                    return (
                                        <g key={`${udl.id}-arrow-${index}`}>
                                            <line
                                                x1={arrowX}
                                                y1={beamY - 64}
                                                x2={arrowX}
                                                y2={beamY - 36}
                                                stroke="#c084fc"
                                                strokeWidth={1.8}
                                            />
                                            <polygon
                                                points={`${arrowX - 4},${beamY - 36} ${arrowX + 4},${beamY - 36} ${arrowX},${beamY - 26}`}
                                                fill="#c084fc"
                                            />
                                        </g>
                                    );
                                })}
                                <text
                                    x={(x1 + x2) / 2}
                                    y={beamY - 82}
                                    textAnchor="middle"
                                    fill="#ede9fe"
                                    fontSize={9}
                                >
                                    {udl.id}
                                </text>
                            </g>
                        );
                    })}

                    {/* Concentrated moments */}
                    {context.moment_loads.map((moment) => {
                        const x = scaleX(moment.position);
                        const radius = 16;
                        const flag = moment.direction === "ccw" ? 1 : 0;
                        return (
                            <g key={moment.id}>
                                <path
                                    d={`M ${x - radius}, ${beamY - 30} A ${radius} ${radius} 0 1 ${flag} ${x + radius}, ${beamY - 30}`}
                                    fill="none"
                                    stroke="#facc15"
                                    strokeWidth={2}
                                />
                                <polygon
                                    points={`${x + (moment.direction === "ccw" ? radius : -radius)},${beamY - 28} ${x + (moment.direction === "ccw" ? radius - 8 : -radius + 8)},${beamY - 20} ${x + (moment.direction === "ccw" ? radius - 8 : -radius + 8)},${beamY - 36}`}
                                    fill="#facc15"
                                />
                                <text x={x} y={beamY - 56} textAnchor="middle" fill="#fef3c7" fontSize={9}>
                                    {moment.id}
                                </text>
                            </g>
                        );
                    })}

                    {/* Beam length annotation */}
                    <text
                        x={width / 2}
                        y={beamY + 60}
                        textAnchor="middle"
                        fill="#94a3b8"
                        fontSize={10}
                    >
                        L = {context.length.toFixed(2)} m
                    </text>
                </svg>
            </div>
        </div>
    );
}
