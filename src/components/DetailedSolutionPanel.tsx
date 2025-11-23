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
        <div className="fixed inset-0 z-50 flex flex-col bg-slate-950 text-slate-100 overflow-hidden">
            {/* Top Bar: Header & Beam Sketch */}
            <div className="flex-none border-b border-slate-800 bg-slate-900/50 backdrop-blur-md">
                <div className="flex items-center justify-between px-6 py-3 border-b border-slate-800/50">
                    <div className="flex items-center gap-4">
                        <button
                            onClick={onClose}
                            className="rounded-full p-2 text-slate-400 hover:bg-slate-800 hover:text-white transition-colors"
                        >
                            <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                            </svg>
                        </button>
                        <div>
                            <h2 className="text-lg font-bold text-white">
                                {viewState === "reactions"
                                    ? "Adım 1: Mesnet Tepkileri"
                                    : viewState === "selection"
                                        ? "Adım 2: Yöntem Seçimi"
                                        : `Adım 3: ${activeMethod?.method_title}`}
                            </h2>
                            <p className="text-xs text-slate-400">
                                {viewState === "selection"
                                    ? "Devam etmek için bir çözüm yöntemi seçiniz."
                                    : `Adım ${currentStepIndex + 1} / ${totalSteps}`}
                            </p>
                        </div>
                    </div>

                    {/* Progress Indicator */}
                    {viewState !== "selection" && (
                        <div className="hidden md:flex gap-1">
                            {Array.from({ length: totalSteps }).map((_, i) => (
                                <div
                                    key={i}
                                    className={clsx(
                                        "h-1.5 rounded-full transition-all",
                                        i === currentStepIndex
                                            ? "w-8 bg-cyan-500"
                                            : i < currentStepIndex
                                                ? "w-4 bg-cyan-500/40"
                                                : "w-4 bg-slate-800"
                                    )}
                                />
                            ))}
                        </div>
                    )}
                </div>

                {/* Always Visible Beam Sketch - REMOVED */}
            </div>

            {/* Main Content Area */}
            <div className="flex-1 overflow-y-auto bg-slate-950">
                <div className="mx-auto max-w-7xl p-6 md:p-10">
                    <AnimatePresence mode="wait">
                        {viewState === "selection" ? (
                            <motion.div
                                key="selection"
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -20 }}
                                className="flex flex-col items-center justify-center"
                            >
                                <h3 className="mb-8 text-2xl font-bold text-white">
                                    Nasıl devam etmek istersiniz?
                                </h3>
                                <div className="grid w-full grid-cols-1 gap-6 md:grid-cols-3">
                                    {solverMethods.map((method) => (
                                        <button
                                            key={method.method_name}
                                            onClick={() => handleMethodSelect(method)}
                                            className={clsx(
                                                "group relative flex flex-col items-start rounded-xl border p-6 text-left transition-all hover:scale-[1.02] hover:shadow-xl",
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
                                    initial={{ opacity: 0, x: 20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: -20 }}
                                    transition={{ duration: 0.3 }}
                                    className="grid grid-cols-1 gap-8 lg:grid-cols-2"
                                >
                                    {/* Left Side: Explanation & Formulas */}
                                    <div className="space-y-6">
                                        <div>
                                            <h3 className="mb-4 text-2xl font-bold text-white">
                                                {currentStep.title}
                                            </h3>
                                            <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6 text-lg leading-relaxed text-slate-300">
                                                {currentStep.explanation}
                                            </div>
                                        </div>

                                        <div className="space-y-4">
                                            {currentStep.general_formula && (
                                                <div className="rounded-xl border border-cyan-500/20 bg-slate-900/30 p-6">
                                                    <p className="mb-3 text-xs font-bold uppercase tracking-wider text-cyan-500">
                                                        Formül
                                                    </p>
                                                    <div className="text-xl overflow-x-auto">
                                                        <BlockMath math={currentStep.general_formula} />
                                                    </div>
                                                </div>
                                            )}
                                            {currentStep.substituted_formula && (
                                                <div className="rounded-xl border border-slate-700/50 bg-slate-900/30 p-6">
                                                    <p className="mb-3 text-xs font-bold uppercase tracking-wider text-slate-500">
                                                        Hesaplama
                                                    </p>
                                                    <div className="text-xl overflow-x-auto">
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
                                                        <BlockMath math={currentStep.numerical_result} />
                                                    </p>
                                                </div>
                                            )}
                                        </div>
                                    </div>

                                    {/* Right Side: Visualizations (Area Method Diagrams etc.) */}
                                    <div className="hidden lg:flex flex-col sticky top-6 self-start space-y-4">
                                        {/* Beam Sketch Container */}
                                        {beamContext && (
                                            <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4 relative">
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

                                                {/* Highlight Overlay for Sections */}
                                                {currentStep?.beam_section && (
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
                                                                    {/* Cut Line Indicator for Section Method */}
                                                                    {activeMethod?.method_name === "section_method" && (
                                                                        <>
                                                                            {/* Dashed cut line in the middle of the section */}
                                                                            {!isPoint && (
                                                                                <div className="absolute top-0 bottom-0 left-1/2 w-0.5 -translate-x-1/2 border-l-2 border-dashed border-cyan-400/60"></div>
                                                                            )}

                                                                            <div className="absolute -top-2 left-1/2 -translate-x-1/2 flex flex-col items-center z-20">
                                                                                <div className="h-4 w-0.5 bg-cyan-400"></div>
                                                                                <div className="text-[10px] font-bold text-cyan-400 bg-slate-900 px-1 rounded border border-cyan-500/30 whitespace-nowrap shadow-lg">
                                                                                    {isPoint ? "KESİM" : "KESİT x"}
                                                                                </div>
                                                                            </div>
                                                                        </>
                                                                    )}

                                                                    {currentStep.beam_section.label && !isPoint && activeMethod?.method_name !== "section_method" && (
                                                                        <div className="absolute -top-6 left-1/2 -translate-x-1/2 whitespace-nowrap rounded bg-cyan-500/20 px-2 py-1 text-xs font-bold text-cyan-300 backdrop-blur-sm border border-cyan-500/30">
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

                                        {/* Diagram Container */}
                                        {currentStep.area_visualization && detailedSolution.diagram && (
                                            <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-3">
                                                <AreaMethodDiagram
                                                    diagram={detailedSolution.diagram}
                                                    visualization={currentStep.area_visualization}
                                                />
                                            </div>
                                        )}
                                    </div>
                                </motion.div>
                            )
                        )}
                    </AnimatePresence>
                </div>
            </div>

            {/* Footer Navigation */}
            <div className="flex-none border-t border-slate-800 bg-slate-900 px-8 py-4">
                <div className="mx-auto flex max-w-7xl items-center justify-between">
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
