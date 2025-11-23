"use client";

import { useEffect, useState } from "react";
import { BlockMath } from "react-katex";
import clsx from "clsx";
import { motion, AnimatePresence } from "framer-motion";
import type { DetailedSolution, SolutionMethod, SupportReaction } from "@/types/beam";
import { BeamSketch } from "./BeamSketch";
import { AreaMethodDiagram } from "./AreaMethodDiagram";

interface DetailedSolutionPanelProps {
    detailedSolution: DetailedSolution;
    reactions?: SupportReaction[];
    isOpen: boolean;
    onClose: () => void;
}

type ViewState = "reactions" | "selection" | "method";

export function DetailedSolutionPanel({
    detailedSolution,
    reactions,
    isOpen,
    onClose,
}: DetailedSolutionPanelProps) {
    const [viewState, setViewState] = useState<ViewState>("reactions");
    const [currentStepIndex, setCurrentStepIndex] = useState(0);
    const [selectedMethodIndex, setSelectedMethodIndex] = useState<number | null>(null);

    // Find the support reactions method (usually the first one)
    const reactionsMethod = detailedSolution.methods.find(
        (m) => m.method_name === "support_reactions"
    );

    // Other methods for selection
    const solverMethods = detailedSolution.methods.filter(
        (m) => m.method_name !== "support_reactions"
    );

    // Reset state when opening
    useEffect(() => {
        if (isOpen) {
            setViewState("reactions");
            setCurrentStepIndex(0);
            setSelectedMethodIndex(null);
            document.body.style.overflow = "hidden";
        } else {
            document.body.style.overflow = "";
        }
        return () => {
            document.body.style.overflow = "";
        };
    }, [isOpen]);

    // Handle ESC
    useEffect(() => {
        if (!isOpen) return;
        const handleEsc = (e: KeyboardEvent) => {
            if (e.key === "Escape") onClose();
        };
        window.addEventListener("keydown", handleEsc);
        return () => window.removeEventListener("keydown", handleEsc);
    }, [isOpen, onClose]);

    if (!isOpen || !reactionsMethod) return null;

    const activeMethod =
        viewState === "reactions"
            ? reactionsMethod
            : viewState === "method" && selectedMethodIndex !== null
                ? detailedSolution.methods[selectedMethodIndex]
                : null;

    const totalSteps = activeMethod ? activeMethod.steps.length : 0;
    const currentStep = activeMethod ? activeMethod.steps[currentStepIndex] : null;
    const beamContext = detailedSolution.beam_context;

    const handleNext = () => {
        if (viewState === "reactions") {
            if (currentStepIndex < totalSteps - 1) {
                setCurrentStepIndex((prev) => prev + 1);
            } else {
                setViewState("selection");
            }
        } else if (viewState === "method") {
            if (currentStepIndex < totalSteps - 1) {
                setCurrentStepIndex((prev) => prev + 1);
            }
        }
    };

    const handlePrev = () => {
        if (viewState === "reactions") {
            if (currentStepIndex > 0) {
                setCurrentStepIndex((prev) => prev - 1);
            }
        } else if (viewState === "selection") {
            setViewState("reactions");
            setCurrentStepIndex(reactionsMethod.steps.length - 1);
        } else if (viewState === "method") {
            if (currentStepIndex > 0) {
                setCurrentStepIndex((prev) => prev - 1);
            } else {
                setViewState("selection");
                setSelectedMethodIndex(null);
            }
        }
    };

    const handleMethodSelect = (method: SolutionMethod) => {
        const index = detailedSolution.methods.findIndex(
            (m) => m.method_name === method.method_name
        );
        setSelectedMethodIndex(index);
        setViewState("method");
        setCurrentStepIndex(0);
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/90 backdrop-blur-md">
            <div className="relative flex h-full w-full flex-col overflow-hidden bg-slate-900 text-slate-100 md:h-[90vh] md:w-[90vw] md:rounded-2xl md:border md:border-slate-800 md:shadow-2xl">
                {/* Header */}
                <div className="flex items-center justify-between border-b border-slate-800 bg-slate-900 px-6 py-4">
                    <div>
                        <h2 className="text-xl font-bold text-cyan-400">
                            {viewState === "reactions"
                                ? "Adım 1: Mesnet Tepkileri"
                                : viewState === "selection"
                                    ? "Adım 2: Yöntem Seçimi"
                                    : `Adım 3: ${activeMethod?.method_title}`}
                        </h2>
                        <p className="text-sm text-slate-400">
                            {viewState === "selection"
                                ? "Devam etmek için bir çözüm yöntemi seçiniz."
                                : `Adım ${currentStepIndex + 1} / ${totalSteps}`}
                        </p>
                    </div>
                    <button
                        onClick={onClose}
                        className="rounded-full p-2 text-slate-400 hover:bg-slate-800 hover:text-white"
                    >
                        <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M6 18L18 6M6 6l12 12"
                            />
                        </svg>
                    </button>
                </div>

                {/* Main Content Area */}
                <div className="flex-1 overflow-hidden relative">
                    <AnimatePresence mode="wait">
                        {viewState === "selection" ? (
                            <motion.div
                                key="selection"
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -20 }}
                                className="flex h-full flex-col items-center justify-center p-8"
                            >
                                <h3 className="mb-8 text-3xl font-bold text-white">
                                    Nasıl devam etmek istersiniz?
                                </h3>
                                <div className="grid w-full max-w-5xl grid-cols-1 gap-6 md:grid-cols-3">
                                    {solverMethods.map((method) => (
                                        <button
                                            key={method.method_name}
                                            onClick={() => handleMethodSelect(method)}
                                            className={clsx(
                                                "group relative flex flex-col items-start rounded-xl border p-6 text-left transition-all hover:scale-105 hover:shadow-xl",
                                                method.recommended
                                                    ? "border-cyan-500/50 bg-cyan-500/10 hover:bg-cyan-500/20"
                                                    : "border-slate-700 bg-slate-800/50 hover:bg-slate-800"
                                            )}
                                        >
                                            {method.recommended && (
                                                <span className="absolute -top-3 right-4 rounded-full bg-cyan-500 px-3 py-1 text-xs font-bold text-slate-900 shadow-lg">
                                                    ÖNERİLEN
                                                </span>
                                            )}
                                            <h4 className="mb-2 text-xl font-bold text-slate-100 group-hover:text-cyan-300">
                                                {method.method_title}
                                            </h4>
                                            <p className="mb-4 text-sm text-slate-400">
                                                {method.description}
                                            </p>
                                            {method.recommendation_reason && (
                                                <div className="mt-auto rounded bg-slate-900/50 p-3 text-xs text-cyan-200/80">
                                                    <span className="font-bold text-cyan-500">Neden? </span>
                                                    {method.recommendation_reason}
                                                </div>
                                            )}
                                        </button>
                                    ))}
                                </div>
                            </motion.div>
                        ) : (
                            currentStep && (
                                <motion.div
                                    key={`${viewState}-${currentStep.step_number}`}
                                    initial={{ opacity: 0, x: 50 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: -50 }}
                                    transition={{ duration: 0.3 }}
                                    className="flex h-full flex-col gap-8 overflow-y-auto p-8 md:flex-row"
                                >
                                    {/* Left Side: Text & Formulas */}
                                    <div className="flex flex-1 flex-col justify-center space-y-8">
                                        <div>
                                            <h3 className="mb-4 text-3xl font-bold text-white">
                                                {currentStep.title}
                                            </h3>
                                            <div className="rounded-2xl border border-slate-700 bg-slate-800/50 p-6 text-lg leading-relaxed text-slate-200 shadow-sm">
                                                {currentStep.explanation}
                                            </div>
                                        </div>

                                        <div className="space-y-4">
                                            {currentStep.general_formula && (
                                                <div className="rounded-xl border border-cyan-500/20 bg-slate-950/50 p-6">
                                                    <p className="mb-2 text-xs font-bold uppercase tracking-wider text-cyan-500">
                                                        Formül
                                                    </p>
                                                    <div className="text-xl">
                                                        <BlockMath math={currentStep.general_formula} />
                                                    </div>
                                                </div>
                                            )}
                                            {currentStep.substituted_formula && (
                                                <div className="rounded-xl border border-slate-700/50 bg-slate-950/30 p-6">
                                                    <p className="mb-2 text-xs font-bold uppercase tracking-wider text-slate-500">
                                                        Hesaplama
                                                    </p>
                                                    <div className="text-xl">
                                                        <BlockMath math={currentStep.substituted_formula} />
                                                    </div>
                                                </div>
                                            )}
                                            {currentStep.numerical_result && (
                                                <div className="rounded-xl border border-green-500/30 bg-green-500/10 p-6">
                                                    <p className="mb-2 text-xs font-bold uppercase tracking-wider text-green-400">
                                                        Sonuç
                                                    </p>
                                                    <p className="text-2xl font-bold text-green-300">
                                                        {currentStep.numerical_result}
                                                    </p>
                                                </div>
                                            )}
                                        </div>
                                    </div>

                                    {/* Right Side: Visuals */}
                                    <div className="flex flex-1 flex-col items-center justify-center gap-6 rounded-3xl bg-slate-950/50 p-4 md:p-8 overflow-hidden relative">
                                        {beamContext && (
                                            <div className="w-full relative">
                                                <div className="pointer-events-none select-none">
                                                    <BeamSketch
                                                        length={beamContext.length}
                                                        supports={beamContext.supports}
                                                        pointLoads={beamContext.point_loads.map((l) => ({ ...l, angleDeg: l.angle_deg }))}
                                                        udls={beamContext.udls}
                                                        momentLoads={beamContext.moment_loads}
                                                        reactions={reactions}
                                                        onSupportPositionChange={() => { }}
                                                        onPointLoadPositionChange={() => { }}
                                                        onUdlRangeChange={() => { }}
                                                        onMomentPositionChange={() => { }}
                                                        onOpenContextMenu={() => { }}
                                                    />
                                                </div>

                                                {/* Highlight Overlay */}
                                                {currentStep.beam_section && (
                                                    <div className="absolute inset-0 pointer-events-none z-10">
                                                        {(() => {
                                                            const start = Math.min(currentStep.beam_section.start, currentStep.beam_section.end);
                                                            const end = Math.max(currentStep.beam_section.start, currentStep.beam_section.end);
                                                            const length = Math.max(beamContext.length, 1e-6);

                                                            const startPct = (start / length) * 100;
                                                            const endPct = (end / length) * 100;

                                                            // BeamSketch uses 2% padding on sides and 96% width for the beam content
                                                            const left = 2 + startPct * 0.96;
                                                            const width = (endPct - startPct) * 0.96;

                                                            // If it's a point cut (width ~ 0)
                                                            const isPoint = width < 0.1;

                                                            return (
                                                                <div
                                                                    className="absolute top-0 bottom-0 border-x-2 border-cyan-400/50 bg-cyan-400/10 transition-all duration-500"
                                                                    style={{
                                                                        left: `${left}%`,
                                                                        width: isPoint ? "2px" : `${width}%`,
                                                                        transform: isPoint ? "translateX(-1px)" : "none",
                                                                    }}
                                                                >
                                                                    {currentStep.beam_section.label && (
                                                                        <div className="absolute -top-6 left-1/2 -translate-x-1/2 whitespace-nowrap rounded bg-cyan-500/20 px-2 py-1 text-xs font-bold text-cyan-300 backdrop-blur-sm">
                                                                            {currentStep.beam_section.label}
                                                                        </div>
                                                                    )}
                                                                </div>
                                                            );
                                                        })()}
                                                    </div>
                                                )}
                                            </div>
                                        )}

                                        {currentStep.area_visualization && detailedSolution.diagram && (
                                            <div className="w-full h-full flex items-center justify-center">
                                                <div className="w-full max-w-full rounded-xl border border-slate-800 bg-slate-900 p-4">
                                                    <AreaMethodDiagram
                                                        diagram={detailedSolution.diagram}
                                                        visualization={currentStep.area_visualization}
                                                    />
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                </motion.div>
                            )
                        )}
                    </AnimatePresence>
                </div>

                {/* Footer Navigation */}
                <div className="flex items-center justify-between border-t border-slate-800 bg-slate-900 px-8 py-6">
                    <button
                        onClick={handlePrev}
                        disabled={viewState === "reactions" && currentStepIndex === 0}
                        className="flex items-center gap-2 rounded-lg px-6 py-3 font-semibold text-slate-400 transition hover:bg-slate-800 hover:text-white disabled:opacity-50"
                    >
                        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                        </svg>
                        Geri
                    </button>

                    {/* Progress Dots */}
                    {viewState !== "selection" && (
                        <div className="flex gap-2">
                            {Array.from({ length: totalSteps }).map((_, i) => (
                                <div
                                    key={i}
                                    className={clsx(
                                        "h-2 rounded-full transition-all",
                                        i === currentStepIndex
                                            ? "w-8 bg-cyan-500"
                                            : i < currentStepIndex
                                                ? "w-2 bg-cyan-500/50"
                                                : "w-2 bg-slate-700"
                                    )}
                                />
                            ))}
                        </div>
                    )}

                    <button
                        onClick={handleNext}
                        disabled={viewState === "method" && currentStepIndex === totalSteps - 1}
                        className={clsx(
                            "flex items-center gap-2 rounded-lg px-8 py-3 font-bold text-white shadow-lg transition-all",
                            viewState === "method" && currentStepIndex === totalSteps - 1
                                ? "cursor-not-allowed bg-slate-700 opacity-50"
                                : "bg-cyan-600 hover:bg-cyan-500 hover:shadow-cyan-500/25"
                        )}
                    >
                        {viewState === "reactions" && currentStepIndex === totalSteps - 1
                            ? "Yöntem Seç"
                            : viewState === "method" && currentStepIndex === totalSteps - 1
                                ? "Tamamlandı"
                                : "İleri"}
                        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                    </button>
                </div>
            </div>
        </div>
    );
}
