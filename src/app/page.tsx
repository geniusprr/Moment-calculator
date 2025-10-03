"use client";

import clsx from "clsx";
import { useCallback, useEffect, useMemo, useState, useTransition } from "react";

import { BeamDiagrams } from "@/components/BeamDiagrams";
import { BeamForm } from "@/components/BeamForm";
import { BeamSketch, SketchContextTarget } from "@/components/BeamSketch";
import { DerivationSteps } from "@/components/DerivationSteps";
import { ResultsPanel } from "@/components/ResultsPanel";
import { solveBeam } from "@/lib/api";
import type {
  BeamSolveRequest,
  BeamSolveResponse,
  MomentLoadInput,
  PointLoadInput,
  SupportInput,
  SupportReaction,
  UdlInput,
} from "@/types/beam";

type PresetConfig = {
  key: string;
  title: string;
  description: string;
  config: {
    length: number;
    supports: SupportInput[];
    pointLoads: PointLoadInput[];
    udls: UdlInput[];
    momentLoads: MomentLoadInput[];
    samplingPoints: number;
  };
};

const PRESETS: PresetConfig[] = [
  {
    key: "default",
    title: "Basit Kiriş",
    description: "10 m açıklıklı basit mesnetli kiriş",
    config: {
      length: 10,
      supports: [
        { id: "A", type: "pin", position: 0 },
        { id: "B", type: "roller", position: 10 },
      ],
      pointLoads: [],
      udls: [],
      momentLoads: [],
      samplingPoints: 401,
    },
  },
  {
    key: "uniform-load",
    title: "Düzgün Yayılı Yük",
    description: "8 m açıklık, tam uzunlukta düzgün yayılı yük",
    config: {
      length: 8,
      supports: [
        { id: "A", type: "pin", position: 0 },
        { id: "B", type: "roller", position: 8 },
      ],
      pointLoads: [],
      udls: [{ id: "Q1", magnitude: 4, start: 0, end: 8, direction: "down" }],
      momentLoads: [],
      samplingPoints: 401,
    },
  },
  {
    key: "combined",
    title: "Karma Yükler",
    description: "10 m kiriş, merkezi nokta yükü ve kısmi düzgün yayılı yük",
    config: {
      length: 10,
      supports: [
        { id: "A", type: "pin", position: 0 },
        { id: "B", type: "roller", position: 10 },
      ],
      pointLoads: [{ id: "F1", magnitude: 15, position: 5, angleDeg: -90 }],
      udls: [{ id: "Q1", magnitude: 5, start: 6, end: 10, direction: "down" }],
      momentLoads: [],
      samplingPoints: 401,
    },
  },
];

const DEFAULT_PRESET = PRESETS[0];

const clampValue = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

const cloneSupports = (items: SupportInput[]): SupportInput[] => items.map((item) => ({ ...item }));
const clonePointLoads = (items: PointLoadInput[]): PointLoadInput[] => items.map((item) => ({ ...item }));
const cloneUdls = (items: UdlInput[]): UdlInput[] => items.map((item) => ({ ...item }));
const cloneMoments = (items: MomentLoadInput[]): MomentLoadInput[] => items.map((item) => ({ ...item }));

function createRandomId(): string {
  const base = typeof crypto !== "undefined" && "randomUUID" in crypto ? crypto.randomUUID() : Math.random().toString(36);
  return base.slice(0, 6).toUpperCase();
}

export default function HomePage() {
  const [length, setLength] = useState(DEFAULT_PRESET.config.length);
  const [supports, setSupports] = useState<SupportInput[]>(cloneSupports(DEFAULT_PRESET.config.supports));
  const [pointLoads, setPointLoads] = useState<PointLoadInput[]>(clonePointLoads(DEFAULT_PRESET.config.pointLoads));
  const [udls, setUdls] = useState<UdlInput[]>(cloneUdls(DEFAULT_PRESET.config.udls));
  const [momentLoads, setMomentLoads] = useState<MomentLoadInput[]>(cloneMoments(DEFAULT_PRESET.config.momentLoads));
  const [samplingPoints, setSamplingPoints] = useState(DEFAULT_PRESET.config.samplingPoints);
  const [result, setResult] = useState<BeamSolveResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();
  const [activePreset, setActivePreset] = useState<string | null>(DEFAULT_PRESET.key);
  const [pendingPresetKey, setPendingPresetKey] = useState<string | null>(DEFAULT_PRESET.key);
  const [contextMenu, setContextMenu] = useState<{ target: SketchContextTarget; clientX: number; clientY: number } | null>(
    null,
  );

  useEffect(() => {
    setSupports((current) => current.map((support) => ({ ...support, position: clampValue(support.position, 0, length) })));
    setPointLoads((current) => current.map((load) => ({ ...load, position: clampValue(load.position, 0, length) })));
    setUdls((current) =>
      current.map((load) => {
        const start = clampValue(load.start, 0, length);
        const end = clampValue(load.end, start + 0.1, length);
        return { ...load, start, end };
      }),
    );
    setMomentLoads((current) => current.map((moment) => ({ ...moment, position: clampValue(moment.position, 0, length) })));
  }, [length]);

  useEffect(() => {
    if (!contextMenu) {
      return;
    }
    const handleEsc = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setContextMenu(null);
      }
    };
    window.addEventListener("keydown", handleEsc);
    return () => window.removeEventListener("keydown", handleEsc);
  }, [contextMenu]);

  const sanitizedSupports = useMemo(
    () =>
      supports
        .filter((support) => support.id.trim().length > 0)
        .map((support) => ({ ...support, id: support.id.toUpperCase(), position: Number(support.position) }))
        .sort((a, b) => a.position - b.position),
    [supports],
  );

  const sanitizedPointLoads = useMemo(
    () =>
      pointLoads
        .filter((load) => load.magnitude > 0)
        .map((load) => ({
          ...load,
          id: load.id.toUpperCase(),
          magnitude: Number(load.magnitude),
          position: Number(load.position),
          angleDeg: Number(load.angleDeg),
        })),
    [pointLoads],
  );

  const sanitizedUdls = useMemo(
    () =>
      udls
        .filter((load) => load.magnitude > 0 && load.end > load.start)
        .map((load) => ({
          ...load,
          id: load.id.toUpperCase(),
          magnitude: Number(load.magnitude),
          start: Number(load.start),
          end: Number(load.end),
          direction: load.direction,
        })),
    [udls],
  );

  const sanitizedMoments = useMemo(
    () =>
      momentLoads
        .filter((load) => load.magnitude > 0)
        .map((load) => ({
          ...load,
          id: load.id.toUpperCase(),
          magnitude: Number(load.magnitude),
          position: Number(load.position),
          direction: load.direction,
        })),
    [momentLoads],
  );

  const disableSolveReason = useMemo(() => {
    if (sanitizedSupports.length !== 2) {
      return "Static solution requires exactly two supports.";
    }
    if (Math.abs(sanitizedSupports[0].position - sanitizedSupports[1].position) < 1e-6) {
      return "Support positions must be distinct.";
    }
    return null;
  }, [sanitizedSupports]);

  const runSolve = useCallback(() => {
    if (disableSolveReason) {
      setResult(null);
      setError(disableSolveReason);
      return;
    }

    const payload: BeamSolveRequest = {
      length,
      supports: sanitizedSupports.map((support) => ({ id: support.id, type: support.type, position: support.position })),
      point_loads: sanitizedPointLoads.map((load) => ({
        id: load.id,
        magnitude: load.magnitude,
        position: load.position,
        angle_deg: load.angleDeg,
      })),
      udls: sanitizedUdls.map((load) => ({
        id: load.id,
        magnitude: load.magnitude,
        start: load.start,
        end: load.end,
        direction: load.direction,
      })),
      moment_loads: sanitizedMoments.map((moment) => ({
        id: moment.id,
        magnitude: moment.magnitude,
        position: moment.position,
        direction: moment.direction,
      })),
      sampling: { points: samplingPoints },
    };

    startTransition(async () => {
      try {
        setError(null);
        const response = await solveBeam(payload);
        setResult(response);
      } catch (err) {
        setResult(null);
        setError(err instanceof Error ? err.message : "Unexpected solver error.");
      }
    });
  }, [disableSolveReason, length, samplingPoints, sanitizedSupports, sanitizedPointLoads, sanitizedUdls, sanitizedMoments]);

  useEffect(() => {
    if (!pendingPresetKey) {
      return;
    }
    runSolve();
    setPendingPresetKey(null);
  }, [pendingPresetKey, runSolve]);

  const clearPresetSelection = useCallback(() => {
    setActivePreset(null);
    setPendingPresetKey(null);
  }, []);

  const applyPreset = useCallback((preset: PresetConfig) => {
    const { config } = preset;
    setLength(config.length);
    setSupports(cloneSupports(config.supports));
    setPointLoads(clonePointLoads(config.pointLoads));
    setUdls(cloneUdls(config.udls));
    setMomentLoads(cloneMoments(config.momentLoads));
    setSamplingPoints(config.samplingPoints);
    setResult(null);
    setError(null);
    setActivePreset(preset.key);
    setPendingPresetKey(preset.key);
  }, []);

  const handleReset = useCallback(() => {
    applyPreset(DEFAULT_PRESET);
  }, [applyPreset]);

  const setLengthAndClear = useCallback((value: number) => {
    clearPresetSelection();
    setError(null);
    setLength(value);
  }, [clearPresetSelection]);

  const handleSupportChange = useCallback(
    (id: string, field: keyof SupportInput, value: string | number) => {
      clearPresetSelection();
      setError(null);
      setSupports((current) =>
        current.map((support) => {
          if (support.id !== id) {
            return support;
          }
          if (field === "id") {
            return { ...support, id: String(value).toUpperCase() };
          }
          if (field === "type") {
            return { ...support, type: value as SupportInput["type"] };
          }
          if (field === "position") {
            return { ...support, position: clampValue(Number(value), 0, length) };
          }
          return support;
        }),
      );
    },
    [clearPresetSelection, length],
  );

  const handlePointLoadChange = useCallback(
    (id: string, field: keyof PointLoadInput, value: string | number) => {
      clearPresetSelection();
      setError(null);
      setPointLoads((current) =>
        current.map((load) => {
          if (load.id !== id) {
            return load;
          }
          if (field === "id") {
            return { ...load, id: String(value).toUpperCase() };
          }
          if (field === "angleDeg") {
            return { ...load, angleDeg: Number(value) };
          }
          if (field === "magnitude" || field === "position") {
            return { ...load, [field]: Number(value) };
          }
          return load;
        }),
      );
    },
    [clearPresetSelection],
  );

  const handleUdlChange = useCallback(
    (id: string, field: keyof UdlInput, value: string | number) => {
      clearPresetSelection();
      setError(null);
      setUdls((current) =>
        current.map((load) => {
          if (load.id !== id) {
            return load;
          }
          if (field === "id") {
            return { ...load, id: String(value).toUpperCase() };
          }
          if (field === "direction") {
            return { ...load, direction: value as UdlInput["direction"] };
          }
          return { ...load, [field]: Number(value) };
        }),
      );
    },
    [clearPresetSelection],
  );

  const handleMomentChange = useCallback(
    (id: string, field: keyof MomentLoadInput, value: string | number) => {
      clearPresetSelection();
      setError(null);
      setMomentLoads((current) =>
        current.map((moment) => {
          if (moment.id !== id) {
            return moment;
          }
          if (field === "id") {
            return { ...moment, id: String(value).toUpperCase() };
          }
          if (field === "direction") {
            return { ...moment, direction: value as MomentLoadInput["direction"] };
          }
          return { ...moment, [field]: Number(value) };
        }),
      );
    },
    [clearPresetSelection],
  );

  const handleSupportPositionDrag = useCallback(
    (id: string, position: number) => {
      clearPresetSelection();
      setSupports((current) =>
        current.map((support) => (support.id === id ? { ...support, position: clampValue(position, 0, length) } : support)),
      );
    },
    [clearPresetSelection, length],
  );

  const handlePointLoadPositionDrag = useCallback(
    (id: string, position: number) => {
      clearPresetSelection();
      setPointLoads((current) =>
        current.map((load) => (load.id === id ? { ...load, position: clampValue(position, 0, length) } : load)),
      );
    },
    [clearPresetSelection, length],
  );

  const handleUdlRangeDrag = useCallback(
    (id: string, edge: "start" | "end", position: number) => {
      clearPresetSelection();
      setUdls((current) =>
        current.map((load) => {
          if (load.id !== id) {
            return load;
          }
          if (edge === "start") {
            const nextStart = Math.min(clampValue(position, 0, length), load.end - 0.1);
            return { ...load, start: nextStart };
          }
          const nextEnd = Math.max(clampValue(position, 0, length), load.start + 0.1);
          return { ...load, end: nextEnd };
        }),
      );
    },
    [clearPresetSelection, length],
  );

  const handleMomentPositionDrag = useCallback(
    (id: string, position: number) => {
      clearPresetSelection();
      setMomentLoads((current) =>
        current.map((moment) => (moment.id === id ? { ...moment, position: clampValue(position, 0, length) } : moment)),
      );
    },
    [clearPresetSelection, length],
  );

  const handleRemoveSupport = useCallback((id: string) => {
    clearPresetSelection();
    setSupports((current) => current.filter((support) => support.id !== id));
  }, [clearPresetSelection]);

  const handleAddSupport = useCallback(
    (position?: number) => {
      clearPresetSelection();
      setSupports((current) => {
        if (current.length >= 2) {
          return current;
        }
        const nextPosition = position !== undefined ? clampValue(position, 0, length) : current.length === 0 ? 0 : length;
        const type: SupportInput["type"] = current.length === 0 ? "pin" : "roller";
        return [
          ...current,
          {
            id: createRandomId(),
            type,
            position: nextPosition,
          },
        ];
      });
    },
    [clearPresetSelection, length],
  );

  const handleAddPointLoad = useCallback(
    (position?: number) => {
      clearPresetSelection();
      setPointLoads((current) => {
        // Find the next available F number (F1, F2, F3, ...)
        const existingNumbers = current
          .map((load) => {
            const match = load.id.match(/^F(\d+)$/);
            return match ? parseInt(match[1], 10) : 0;
          })
          .filter((num) => num > 0);
        const nextNumber = existingNumbers.length > 0 ? Math.max(...existingNumbers) + 1 : 1;
        return [
          ...current,
          {
            id: `F${nextNumber}`,
            magnitude: 5,
            position: clampValue(position ?? length / 2, 0, length),
            angleDeg: -90,
          },
        ];
      });
    },
    [clearPresetSelection, length],
  );

  const handleRemovePointLoad = useCallback((id: string) => {
    clearPresetSelection();
    setPointLoads((current) => current.filter((load) => load.id !== id));
  }, [clearPresetSelection]);

  const togglePointDirection = useCallback((id: string) => {
    clearPresetSelection();
    setPointLoads((current) =>
      current.map((load) =>
        load.id === id
          ? {
              ...load,
              angleDeg: ((load.angleDeg + 180) % 360) - 180,
            }
          : load,
      ),
    );
  }, [clearPresetSelection]);

  const handleAddUdl = useCallback(
    (center?: number) => {
      clearPresetSelection();
      setUdls((current) => {
        const span = Math.min(Math.max(length * 0.3, 0.5), length);
        // Varsayılan ekleme: kirişin tam ortasına yerleştir
        const mid = clampValue(center ?? length / 2, 0, length);
        const start = clampValue(mid - span / 2, 0, length - 0.1);
        const end = clampValue(start + span, start + 0.1, length);
        return [
          ...current,
          {
            id: createRandomId(),
            magnitude: 3,
            start,
            end,
            direction: "down",
          },
        ];
      });
    },
    [clearPresetSelection, length],
  );

  const handleRemoveUdl = useCallback((id: string) => {
    clearPresetSelection();
    setUdls((current) => current.filter((load) => load.id !== id));
  }, [clearPresetSelection]);

  const toggleUdlDirection = useCallback((id: string) => {
    clearPresetSelection();
    setUdls((current) =>
      current.map((load) =>
        load.id === id
          ? { ...load, direction: load.direction === "down" ? "up" : "down" }
          : load,
      ),
    );
  }, [clearPresetSelection]);

  const handleAddMoment = useCallback(
    (position?: number) => {
      clearPresetSelection();
      setMomentLoads((current) => [
        ...current,
        {
          id: createRandomId(),
          magnitude: 10,
          position: clampValue(position ?? 0, 0, length),
          direction: "ccw",
        },
      ]);
    },
    [clearPresetSelection, length],
  );

  const handleRemoveMoment = useCallback((id: string) => {
    clearPresetSelection();
    setMomentLoads((current) => current.filter((moment) => moment.id !== id));
  }, [clearPresetSelection]);

  const toggleMomentDirection = useCallback((id: string) => {
    clearPresetSelection();
    setMomentLoads((current) =>
      current.map((moment) =>
        moment.id === id
          ? { ...moment, direction: moment.direction === "ccw" ? "cw" : "ccw" }
          : moment,
      ),
    );
  }, [clearPresetSelection]);

  const diagramData = result?.diagram ?? { x: [], shear: [], moment: [], normal: [] };
  const reactions: SupportReaction[] | undefined = result?.reactions;

  const closeContextMenu = useCallback(() => setContextMenu(null), []);

  const handleContextMenuSelection = useCallback(
    (action: () => void) => {
      action();
      closeContextMenu();
    },
    [closeContextMenu],
  );

  const contextMenuItems = useMemo(() => {
    if (!contextMenu) {
      return [] as Array<{ label: string; disabled?: boolean; action: () => void }>;
    }
    const { target } = contextMenu;
    switch (target.kind) {
      case "blank": {
        const canAddSupport = supports.length < 2;
        return [
          {
            label: canAddSupport ? "Add support" : "Add support (two max)",
            disabled: !canAddSupport,
            action: () => handleAddSupport(target.x),
          },
          { label: "Add point load", action: () => handleAddPointLoad(target.x) },
          { label: "Add distributed load", action: () => handleAddUdl(target.x) },
          { label: "Add moment", action: () => handleAddMoment(target.x) },
        ];
      }
      case "support":
        return [
          { label: "Remove support", action: () => handleRemoveSupport(target.id) },
        ];
      case "point":
        return [
          { label: "Flip direction", action: () => togglePointDirection(target.id) },
          { label: "Remove point load", action: () => handleRemovePointLoad(target.id) },
        ];
      case "udl":
        return [
          { label: "Toggle direction", action: () => toggleUdlDirection(target.id) },
          { label: "Remove distributed load", action: () => handleRemoveUdl(target.id) },
        ];
      case "moment":
        return [
          { label: "Toggle direction", action: () => toggleMomentDirection(target.id) },
          { label: "Remove moment", action: () => handleRemoveMoment(target.id) },
        ];
      default:
        return [];
    }
  }, [contextMenu, handleAddMoment, handleAddPointLoad, handleAddSupport, handleAddUdl, handleRemoveMoment, handleRemovePointLoad, handleRemoveSupport, handleRemoveUdl, supports.length, toggleMomentDirection, togglePointDirection, toggleUdlDirection]);

  return (
    <main
      className="pb-16"
      onClick={() => {
        if (contextMenu) {
          closeContextMenu();
        }
      }}
    >
       <div className="mx-auto flex w-full max-w-none flex-col gap-5 px-4 pt-4 sm:px-6">
        <section className="panel space-y-4 p-4 sm:p-5">
          <div className="flex items-center justify-between">
            <span className="tag">Ön Ayarlar</span>
            <button
              type="button"
              onClick={handleReset}
              className="rounded-full border border-slate-700/70 px-3 py-1 text-xs text-slate-300 transition hover:border-slate-500 hover:text-white"
            >
              Varsayılana sıfırla
            </button>
          </div>
          <p className="text-xs text-slate-400">Bir şablondan başlayın veya özel kiriş tasarımınızı geliştirmeye devam edin.</p>
          <div className="flex flex-col gap-3 md:flex-row md:flex-wrap">
            {PRESETS.map((preset) => (
              <button
                key={preset.key}
                type="button"
                onClick={() => applyPreset(preset)}
                className={clsx(
                  "flex-1 rounded-2xl border px-4 py-3 text-left transition",
                  activePreset === preset.key
                    ? "border-cyan-400 bg-cyan-500/10"
                    : "border-slate-700/80 bg-slate-900/60 hover:border-cyan-400",
                )}
              >
                <p className="text-sm font-semibold text-slate-100">{preset.title}</p>
                <p className="text-xs text-slate-400">{preset.description}</p>
              </button>
            ))}
          </div>
        </section>

         {/* BeamSketch moved into center column to sit between model and results */}

         <div className="grid gap-6 xl:grid-cols-[380px_minmax(0,1fr)_380px]">
          <div className="space-y-6">
            <BeamForm
              length={length}
              onLengthChange={setLengthAndClear}
              supports={supports}
              onSupportChange={handleSupportChange}
              onAddSupport={() => handleAddSupport()}
              onRemoveSupport={handleRemoveSupport}
              pointLoads={pointLoads}
              onPointLoadChange={handlePointLoadChange}
              onAddPointLoad={() => handleAddPointLoad()}
              onRemovePointLoad={handleRemovePointLoad}
              udls={udls}
              onUdlChange={handleUdlChange}
              onAddUdl={() => handleAddUdl()}
              onRemoveUdl={handleRemoveUdl}
              momentLoads={momentLoads}
              onMomentChange={handleMomentChange}
              onAddMoment={() => handleAddMoment()}
              onRemoveMoment={handleRemoveMoment}
              samplingPoints={samplingPoints}
              onSamplingChange={(value) => {
                clearPresetSelection();
                const bounded = Math.round(clampValue(value, 101, 801));
                const adjusted = bounded % 2 === 0 ? bounded + 1 : bounded;
                setSamplingPoints(adjusted);
              }}
              onSolve={runSolve}
              onReset={handleReset}
              solving={isPending}
              disableSolveReason={disableSolveReason}
            />
          </div>

           <div className="space-y-6">
             <BeamSketch
               length={length}
               supports={sanitizedSupports}
               pointLoads={sanitizedPointLoads}
               udls={sanitizedUdls}
               momentLoads={sanitizedMoments}
               reactions={reactions}
               onSupportPositionChange={handleSupportPositionDrag}
               onPointLoadPositionChange={handlePointLoadPositionDrag}
               onUdlRangeChange={handleUdlRangeDrag}
               onMomentPositionChange={handleMomentPositionDrag}
               onOpenContextMenu={(target, clientX, clientY) => setContextMenu({ target, clientX, clientY })}
             />
             <BeamDiagrams
               x={diagramData.x}
               shear={diagramData.shear}
               moment={diagramData.moment}
               normal={diagramData.normal}
               loading={isPending}
             />
           </div>

          <div className="space-y-6">
            <ResultsPanel
              reactions={reactions}
              solveTimeMs={result?.meta.solve_time_ms}
              warnings={result?.meta.validation_warnings ?? []}
              error={error}
            />
            <DerivationSteps steps={result?.derivations ?? []} />
          </div>
        </div>
      </div>

      {contextMenu && (
        <div
          className="fixed z-50 min-w-[180px] rounded-xl border border-slate-700/80 bg-slate-900/95 p-2 text-sm text-slate-100 shadow-lg"
          style={{ top: contextMenu.clientY, left: contextMenu.clientX }}
          onClick={(event) => event.stopPropagation()}
        >
          {contextMenuItems.map((item) => (
            <button
              key={item.label}
              type="button"
              disabled={item.disabled}
              onClick={() => handleContextMenuSelection(item.action)}
              className={clsx(
                "flex w-full items-center gap-2 rounded-lg px-3 py-1.5 text-left transition",
                item.disabled
                  ? "cursor-not-allowed text-slate-500"
                  : "hover:bg-slate-800 hover:text-white",
              )}
            >
              {item.label}
            </button>
          ))}
          {contextMenuItems.length === 0 && (
            <div className="px-3 py-1.5 text-xs text-slate-500">No actions available</div>
          )}
        </div>
      )}
    </main>
  );
}
