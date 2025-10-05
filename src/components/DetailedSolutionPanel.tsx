"use client";

import { useEffect, useState } from "react";
import { BlockMath } from "react-katex";
import clsx from "clsx";
import type { DetailedSolution } from "@/types/beam";
import { BeamSectionDiagram } from "./BeamSectionDiagram";
import { AreaMethodDiagram } from "./AreaMethodDiagram";

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
    const beamContext = detailedSolution.beam_context;
    const recommendedMethod = detailedSolution.methods.find(
        (method) => method.recommended && method.method_name !== "support_reactions"
    ) ?? detailedSolution.methods.find((method) => method.recommended);

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
                            {/* Recommendation Overview (only for support reactions) */}
                            {activeMethod.method_name === "support_reactions" && (
                                <div className="panel border border-cyan-500/20 bg-slate-900/70 p-6">
                                    <div className="flex flex-wrap items-center justify-between gap-3">
                                        <div>
                                            <p className="text-xs font-semibold uppercase tracking-wide text-cyan-300">Yöntem Seçim Rehberi</p>
                                            <p className="text-sm text-slate-300">
                                                Hangi yöntemi ne zaman tercih edeceğinizi aşağıdaki özetten görebilirsiniz.
                                            </p>
                                        </div>
                                        {recommendedMethod && (
                                            <span className="rounded-full bg-cyan-500/20 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-cyan-200">
                                                Önerilen: {recommendedMethod.method_title}
                                            </span>
                                        )}
                                    </div>
                                    <div className="mt-4 space-y-3">
                                        {detailedSolution.methods
                                            .filter((method) => method.method_name !== "support_reactions")
                                            .map((method) => (
                                                <div
                                                    key={`summary-${method.method_name}`}
                                                    className={clsx(
                                                        "rounded-lg border px-4 py-3",
                                                        method.recommended
                                                            ? "border-cyan-500/40 bg-cyan-500/10"
                                                            : "border-slate-700/60 bg-slate-800/40"
                                                    )}
                                                >
                                                    <div className="flex items-start justify-between gap-3">
                                                        <div>
                                                            <p className="text-sm font-semibold text-slate-100">{method.method_title}</p>
                                                            {method.recommendation_reason && (
                                                                <p className="mt-1 text-xs leading-relaxed text-slate-300">
                                                                    {method.recommendation_reason}
                                                                </p>
                                                            )}
                                                        </div>
                                                        {method.recommended && (
                                                            <span className="rounded-full bg-cyan-500 px-3 py-1 text-xs font-semibold text-slate-900">
                                                                Tavsiye edilir
                                                            </span>
                                                        )}
                                                    </div>
                                                </div>
                                            ))}
                                    </div>
                                </div>
                            )}

                            {/* Diagram panel removed for shear and area methods as requested */}

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
                                    const hasBeamHighlight = Boolean(step.beam_section && beamContext);
                                    const hasAreaVisualization = Boolean(step.area_visualization && detailedSolution.diagram);

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
                                                <div className="px-6 py-6">
                                                    <div
                                                        className={clsx(
                                                            "flex flex-col gap-6",
                                                            (hasBeamHighlight || hasAreaVisualization) && "lg:flex-row"
                                                        )}
                                                    >
                                                        <div
                                                            className={clsx(
                                                                "space-y-4",
                                                                (hasBeamHighlight || hasAreaVisualization) && "lg:flex-1 lg:pr-4"
                                                            )}
                                                        >
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

                                                        {step.beam_section && beamContext && (
                                                            <div className="shrink-0 w-full lg:w-[340px]">
                                                                <BeamSectionDiagram
                                                                    context={beamContext}
                                                                    highlight={step.beam_section}
                                                                />
                                                            </div>
                                                        )}
                                                        {hasAreaVisualization && detailedSolution.diagram && step.area_visualization && (
                                                            <div className="shrink-0 w-full lg:w-[360px]">
                                                                <AreaMethodDiagram
                                                                    diagram={detailedSolution.diagram}
                                                                    visualization={step.area_visualization}
                                                                />
                                                            </div>
                                                        )}
                                                    </div>
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

// Diagram bileşeni kaldırıldı
