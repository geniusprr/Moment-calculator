"use client";

import { useEffect, useState } from "react";
import { BlockMath } from "react-katex";
import clsx from "clsx";
import type { DetailedSolution } from "@/types/beam";

interface DetailedSolutionPanelProps {
    detailedSolution: DetailedSolution;
    isOpen: boolean;
    onClose: () => void;
}

export function DetailedSolutionPanel({
    detailedSolution,
    isOpen,
    onClose,
}: DetailedSolutionPanelProps) {
    const [activeMethodIndex, setActiveMethodIndex] = useState(0);
    const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set([0]));

    // Handle ESC key to close
    useEffect(() => {
        if (!isOpen) return;

        const handleEsc = (event: KeyboardEvent) => {
            if (event.key === "Escape") {
                onClose();
            }
        };

        window.addEventListener("keydown", handleEsc);
        return () => window.removeEventListener("keydown", handleEsc);
    }, [isOpen, onClose]);

    // Prevent scroll when open
    useEffect(() => {
        if (isOpen) {
            document.body.style.overflow = "hidden";
        } else {
            document.body.style.overflow = "";
        }

        return () => {
            document.body.style.overflow = "";
        };
    }, [isOpen]);

    // Reset to first method when opening
    useEffect(() => {
        if (isOpen) {
            setActiveMethodIndex(0);
            setExpandedSteps(new Set([0]));
        }
    }, [isOpen]);

    const toggleStep = (stepNumber: number) => {
        setExpandedSteps((prev) => {
            const next = new Set(prev);
            if (next.has(stepNumber)) {
                next.delete(stepNumber);
            } else {
                next.add(stepNumber);
            }
            return next;
        });
    };

    const expandAll = () => {
        const allSteps = new Set(
            detailedSolution.methods[activeMethodIndex].steps.map((s) => s.step_number)
        );
        setExpandedSteps(allSteps);
    };

    const collapseAll = () => {
        setExpandedSteps(new Set());
    };

    if (!isOpen) return null;

    const activeMethod = detailedSolution.methods[activeMethodIndex];

    return (
        <>
            {/* Backdrop */}
            <div
                className={clsx(
                    "fixed inset-0 z-40 bg-slate-950/80 backdrop-blur-sm transition-opacity duration-300",
                    isOpen ? "opacity-100" : "opacity-0"
                )}
                onClick={onClose}
            />

            {/* Panel */}
            <div
                className={clsx(
                    "fixed inset-0 z-50 transform transition-transform duration-300 ease-out",
                    isOpen ? "translate-x-0" : "translate-x-full"
                )}
            >
                <div className="flex h-full w-full flex-col bg-slate-900">
                    {/* Header */}
                    <div className="flex items-center justify-between border-b border-slate-800 bg-gradient-to-r from-slate-900 via-slate-800 to-slate-900 px-6 py-4">
                        <div>
                            <h2 className="text-2xl font-bold text-slate-100">Detaylı Çözüm</h2>
                            <p className="text-sm text-slate-400">
                                Farklı yöntemlerle adım adım kiriş analizi
                            </p>
                        </div>
                        <button
                            onClick={onClose}
                            className="flex h-10 w-10 items-center justify-center rounded-full bg-slate-800 text-slate-300 transition hover:bg-slate-700 hover:text-white"
                            aria-label="Kapat"
                        >
                            <svg
                                className="h-6 w-6"
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                            >
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={2}
                                    d="M6 18L18 6M6 6l12 12"
                                />
                            </svg>
                        </button>
                    </div>

                    {/* Method Tabs */}
                    <div className="border-b border-slate-800 bg-slate-900/50 px-6">
                        <div className="flex gap-2 overflow-x-auto">
                            {detailedSolution.methods.map((method, index) => (
                                <button
                                    key={method.method_name}
                                    onClick={() => {
                                        setActiveMethodIndex(index);
                                        setExpandedSteps(new Set([0]));
                                    }}
                                    className={clsx(
                                        "whitespace-nowrap border-b-2 px-4 py-3 text-sm font-medium transition",
                                        activeMethodIndex === index
                                            ? "border-cyan-400 text-cyan-400"
                                            : "border-transparent text-slate-400 hover:border-slate-600 hover:text-slate-200"
                                    )}
                                >
                                    {method.method_title}
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Content */}
                    <div className="flex-1 overflow-y-auto px-6 py-6">
                        <div className="mx-auto max-w-5xl space-y-6">
                            {/* Method Description */}
                            <div className="panel p-6">
                                <div className="mb-2 flex items-center gap-2">
                                    <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-cyan-500/20">
                                        <svg
                                            className="h-5 w-5 text-cyan-400"
                                            fill="none"
                                            stroke="currentColor"
                                            viewBox="0 0 24 24"
                                        >
                                            <path
                                                strokeLinecap="round"
                                                strokeLinejoin="round"
                                                strokeWidth={2}
                                                d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                                            />
                                        </svg>
                                    </div>
                                    <h3 className="text-lg font-semibold text-slate-100">
                                        {activeMethod.method_title}
                                    </h3>
                                </div>
                                <p className="text-slate-300">{activeMethod.description}</p>
                            </div>

                            {/* Diagrams - Show for shear and area methods */}
                            {detailedSolution.diagram && (activeMethod.method_name === "shear" || activeMethod.method_name === "area") && (
                                <div className="panel p-6">
                                    <h4 className="mb-4 text-sm font-semibold uppercase tracking-wide text-slate-400">
                                        Kesme ve Moment Grafikleri
                                    </h4>
                                    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                                        {/* Shear Diagram */}
                                        <div>
                                            <p className="mb-2 text-xs font-medium text-slate-400">Kesme Kuvveti Grafiği</p>
                                            <div className="rounded-lg bg-slate-950/50 p-4">
                                                <MiniDiagram
                                                    data={detailedSolution.diagram.shear}
                                                    xData={detailedSolution.diagram.x}
                                                    color="#06b6d4"
                                                    label="V (kN)"
                                                />
                                            </div>
                                        </div>

                                        {/* Moment Diagram */}
                                        <div>
                                            <p className="mb-2 text-xs font-medium text-slate-400">Eğilme Momenti Grafiği</p>
                                            <div className="rounded-lg bg-slate-950/50 p-4">
                                                <MiniDiagram
                                                    data={detailedSolution.diagram.moment}
                                                    xData={detailedSolution.diagram.x}
                                                    color="#22c55e"
                                                    label="M (kN·m)"
                                                />
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}

                            {/* Controls */}
                            <div className="flex justify-end gap-2">
                                <button
                                    onClick={expandAll}
                                    className="rounded-lg border border-slate-700 bg-slate-800/50 px-3 py-1.5 text-xs text-slate-300 transition hover:bg-slate-700"
                                >
                                    Tümünü Aç
                                </button>
                                <button
                                    onClick={collapseAll}
                                    className="rounded-lg border border-slate-700 bg-slate-800/50 px-3 py-1.5 text-xs text-slate-300 transition hover:bg-slate-700"
                                >
                                    Tümünü Kapat
                                </button>
                            </div>

                            {/* Steps */}
                            <div className="space-y-4">
                                {activeMethod.steps.map((step) => {
                                    const isExpanded = expandedSteps.has(step.step_number);

                                    return (
                                        <div
                                            key={step.step_number}
                                            className="panel overflow-hidden border border-slate-800/60"
                                        >
                                            {/* Step Header */}
                                            <button
                                                onClick={() => toggleStep(step.step_number)}
                                                className="flex w-full items-center justify-between bg-slate-800/30 px-6 py-4 text-left transition hover:bg-slate-800/50"
                                            >
                                                <div className="flex items-center gap-4">
                                                    <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-full bg-cyan-500/20 text-cyan-400 font-semibold">
                                                        {step.step_number}
                                                    </div>
                                                    <div>
                                                        <h4 className="text-base font-semibold text-slate-100">
                                                            {step.title}
                                                        </h4>
                                                        {!isExpanded && (
                                                            <p className="text-xs text-slate-500">
                                                                Detayları görmek için tıklayın
                                                            </p>
                                                        )}
                                                    </div>
                                                </div>
                                                <svg
                                                    className={clsx(
                                                        "h-5 w-5 flex-shrink-0 text-slate-400 transition-transform",
                                                        isExpanded && "rotate-180"
                                                    )}
                                                    fill="none"
                                                    stroke="currentColor"
                                                    viewBox="0 0 24 24"
                                                >
                                                    <path
                                                        strokeLinecap="round"
                                                        strokeLinejoin="round"
                                                        strokeWidth={2}
                                                        d="M19 9l-7 7-7-7"
                                                    />
                                                </svg>
                                            </button>

                                            {/* Step Content */}
                                            {isExpanded && (
                                                <div className="space-y-4 px-6 py-6">
                                                    {/* Explanation */}
                                                    <div>
                                                        <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
                                                            Açıklama
                                                        </p>
                                                        <div className="rounded-lg bg-slate-800/30 border border-slate-700/50 p-4">
                                                            <p className="whitespace-pre-line text-sm leading-relaxed text-slate-200 font-mono">
                                                                {step.explanation}
                                                            </p>
                                                        </div>
                                                    </div>

                                                    {/* Formulas */}
                                                    {(step.general_formula || step.substituted_formula) && (
                                                        <div className="space-y-4">
                                                            {/* General Formula */}
                                                            {step.general_formula && (
                                                                <div>
                                                                    <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-slate-400">
                                                                        Genel Formül
                                                                    </p>
                                                                    <div className="rounded-xl bg-slate-950/50 p-6 border border-cyan-500/20">
                                                                        <BlockMath math={step.general_formula} />
                                                                    </div>
                                                                </div>
                                                            )}

                                                            {/* Substituted Formula */}
                                                            {step.substituted_formula && (
                                                                <div>
                                                                    <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-slate-400">
                                                                        Değerlerin Yerine Konulmuş Hali
                                                                    </p>
                                                                    <div className="rounded-xl bg-slate-950/50 p-6 border border-green-500/20">
                                                                        <BlockMath math={step.substituted_formula} />
                                                                    </div>
                                                                </div>
                                                            )}
                                                        </div>
                                                    )}

                                                    {/* Numerical Result */}
                                                    {step.numerical_result && (
                                                        <div>
                                                            <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
                                                                Sonuç
                                                            </p>
                                                            <div className="rounded-lg bg-green-500/10 border border-green-500/30 px-4 py-3">
                                                                <p className="text-sm font-medium text-green-300">
                                                                    {step.numerical_result}
                                                                </p>
                                                            </div>
                                                        </div>
                                                    )}
                                                </div>
                                            )}
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </>
    );
}

// Mini Diagram Component for inline visualization
function MiniDiagram({ data, xData, color, label }: { data: number[], xData: number[], color: string, label: string }) {
    const width = 400;
    const height = 150;
    const padding = 30;

    // Find min/max for scaling
    const minX = Math.min(...xData);
    const maxX = Math.max(...xData);
    const minY = Math.min(...data, 0);
    const maxY = Math.max(...data, 0);

    // Scale functions
    const scaleX = (x: number) => padding + ((x - minX) / (maxX - minX)) * (width - 2 * padding);
    const scaleY = (y: number) => {
        const range = maxY - minY;
        if (range === 0) return height / 2;
        return height - padding - ((y - minY) / range) * (height - 2 * padding);
    };

    // Create path
    const pathData = data.map((y, i) => {
        const x = scaleX(xData[i]);
        const yPos = scaleY(y);
        return `${i === 0 ? 'M' : 'L'} ${x} ${yPos}`;
    }).join(' ');

    // Zero line position
    const zeroY = scaleY(0);

    return (
        <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} className="overflow-visible">
            {/* Grid lines */}
            <line x1={padding} y1={zeroY} x2={width - padding} y2={zeroY} stroke="#475569" strokeWidth="1" strokeDasharray="4 2" />

            {/* Axes */}
            <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#64748b" strokeWidth="1.5" />
            <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#64748b" strokeWidth="1.5" />

            {/* Data path */}
            <path d={pathData} fill="none" stroke={color} strokeWidth="2" />

            {/* Fill area if applicable */}
            {data.some(v => v !== 0) && (
                <path
                    d={`M ${scaleX(xData[0])} ${zeroY} ${pathData.replace('M', 'L')} L ${scaleX(xData[xData.length - 1])} ${zeroY} Z`}
                    fill={color}
                    fillOpacity="0.1"
                />
            )}

            {/* Labels */}
            <text x={padding} y={padding - 10} fill="#94a3b8" fontSize="10" fontFamily="monospace">
                {label}
            </text>
            <text x={width - padding} y={height - 10} fill="#94a3b8" fontSize="10" textAnchor="end" fontFamily="monospace">
                x (m)
            </text>

            {/* Min/Max values */}
            {maxY !== 0 && (
                <text x={padding - 5} y={scaleY(maxY) + 3} fill="#94a3b8" fontSize="9" textAnchor="end" fontFamily="monospace">
                    {maxY.toFixed(1)}
                </text>
            )}
            {minY !== 0 && (
                <text x={padding - 5} y={scaleY(minY) + 3} fill="#94a3b8" fontSize="9" textAnchor="end" fontFamily="monospace">
                    {minY.toFixed(1)}
                </text>
            )}
        </svg>
    );
}
