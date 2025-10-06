"use client";

import clsx from "clsx";
import Image from "next/image";
import { useCallback, useEffect, useMemo, useState, useTransition } from "react";

import { BeamDiagrams } from "@/components/BeamDiagrams";
import { BeamForm } from "@/components/BeamForm";
import { BeamSketch, SketchContextTarget } from "@/components/BeamSketch";
import { DetailedSolutionPanel } from "@/components/DetailedSolutionPanel";
import { solveBeam } from "@/lib/api";
import KtoLogo from "../../assets/KtoLOGO.png";
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
  };
};

type ContextMenuItem = {
  label: string;
  disabled?: boolean;
  action?: () => void;
  tooltip?: string;
  submenu?: ContextMenuItem[];
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
      udls: [{ id: "Q1", magnitude: 4, start: 0, end: 8, direction: "down", shape: "uniform" }],
      momentLoads: [],
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
      udls: [{ id: "Q1", magnitude: 5, start: 6, end: 10, direction: "down", shape: "uniform" }],
      momentLoads: [],
    },
  },
];

const DEFAULT_PRESET = PRESETS[0];

const clampValue = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

const normalizeAngle = (angle: number) => {
  const wrapped = ((angle % 360) + 360) % 360;
  return wrapped > 180 ? wrapped - 360 : wrapped;
};

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
  const [result, setResult] = useState<BeamSolveResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();
  const [activePreset, setActivePreset] = useState<string | null>(DEFAULT_PRESET.key);
  const [pendingPresetKey, setPendingPresetKey] = useState<string | null>(DEFAULT_PRESET.key);
  const [contextMenu, setContextMenu] = useState<{ target: SketchContextTarget; clientX: number; clientY: number } | null>(
    null,
  );
  const [openSubmenu, setOpenSubmenu] = useState<string | null>(null);
  const [isDetailedSolutionOpen, setIsDetailedSolutionOpen] = useState(false);

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

  useEffect(() => {
    setOpenSubmenu(null);
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
          angleDeg: normalizeAngle(Number(load.angleDeg)),
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
          shape: load.shape,
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
        shape: load.shape,
      })),
      moment_loads: sanitizedMoments.map((moment) => ({
        id: moment.id,
        magnitude: moment.magnitude,
        position: moment.position,
        direction: moment.direction,
      })),
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
  }, [disableSolveReason, length, sanitizedSupports, sanitizedPointLoads, sanitizedUdls, sanitizedMoments]);

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
            return { ...load, angleDeg: normalizeAngle(Number(value)) };
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
          if (field === "shape") {
            return { ...load, shape: value as UdlInput["shape"] };
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

  const handlePointLoadMagnitudeChange = useCallback(
    (id: string, magnitude: number) => {
      handlePointLoadChange(id, "magnitude", magnitude);
    },
    [handlePointLoadChange],
  );

  const handleUdlMagnitudeChange = useCallback(
    (id: string, magnitude: number) => {
      handleUdlChange(id, "magnitude", magnitude);
    },
    [handleUdlChange],
  );

  const handleMomentMagnitudeChange = useCallback(
    (id: string, magnitude: number) => {
      handleMomentChange(id, "magnitude", magnitude);
    },
    [handleMomentChange],
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
        const usedIds = new Set(current.map((support) => support.id.toUpperCase()));
        const preferredIds = ["A", "B"] as const;
        const nextId = preferredIds.find((candidate) => !usedIds.has(candidate)) ?? createRandomId();
        return [
          ...current,
          {
            id: nextId,
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

  const rotatePointAngle = useCallback((id: string, delta: number) => {
    clearPresetSelection();
    setPointLoads((current) =>
      current.map((load) =>
        load.id === id
          ? {
            ...load,
            angleDeg: normalizeAngle(load.angleDeg + delta),
          }
          : load,
      ),
    );
  }, [clearPresetSelection]);

  const setPointAngle = useCallback((id: string, nextAngle: number) => {
    clearPresetSelection();
    setPointLoads((current) =>
      current.map((load) => (load.id === id ? { ...load, angleDeg: normalizeAngle(nextAngle) } : load)),
    );
  }, [clearPresetSelection]);

  const promptPointAngle = useCallback((id: string) => {
    const current = pointLoads.find((load) => load.id === id)?.angleDeg ?? 0;
    const input = window.prompt("Yük açısını (°) girin", current.toString());
    if (input === null) {
      return;
    }
    const normalizedInput = input.replace(",", ".");
    const value = Number(normalizedInput.trim());
    if (!Number.isFinite(value)) {
      window.alert("Geçersiz açı değeri.");
      return;
    }
    setPointAngle(id, value);
  }, [pointLoads, setPointAngle]);

  const handleAddUdl = useCallback(
    (center?: number) => {
      clearPresetSelection();
      setUdls((current) => {
        const usedNumbers = new Set(
          current
            .map((item) => {
              const match = item.id.match(/^Q(\d+)$/i);
              return match ? Number.parseInt(match[1], 10) : null;
            })
            .filter((value): value is number => value !== null),
        );
        let nextIndex = 1;
        while (usedNumbers.has(nextIndex)) {
          nextIndex += 1;
        }
        const span = Math.min(Math.max(length * 0.3, 0.5), length);
        // Varsayılan ekleme: kirişin tam ortasına yerleştir
        const mid = clampValue(center ?? length / 2, 0, length);
        const start = clampValue(mid - span / 2, 0, length - 0.1);
        const end = clampValue(start + span, start + 0.1, length);
        return [
          ...current,
          {
            id: `Q${nextIndex}`,
            magnitude: 3,
            start,
            end,
            direction: "down",
            shape: "uniform",
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

  const setUdlShape = useCallback((id: string, shape: UdlInput["shape"]) => {
    clearPresetSelection();
    setUdls((current) =>
      current.map((load) => (load.id === id ? { ...load, shape } : load)),
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

  const closeContextMenu = useCallback(() => {
    setOpenSubmenu(null);
    setContextMenu(null);
  }, []);

  const handleContextMenuSelection = useCallback(
    (action?: () => void) => {
      if (action) {
        action();
      }
      closeContextMenu();
    },
    [closeContextMenu],
  );

  const contextMenuItems = useMemo<ContextMenuItem[]>(() => {
    if (!contextMenu) {
      return [];
    }
    const { target } = contextMenu;
    switch (target.kind) {
      case "blank": {
        const canAddSupport = supports.length < 2;
        return [
          {
            label: canAddSupport ? "Mesnet ekle" : "Mesnet ekle (en fazla iki)",
            disabled: !canAddSupport,
            action: () => handleAddSupport(target.x),
          },
          { label: "Tekil yük ekle", action: () => handleAddPointLoad(target.x) },
          { label: "Yayılı yük ekle", action: () => handleAddUdl(target.x) },
          { label: "Moment ekle", action: () => handleAddMoment(target.x) },
        ];
      }
      case "support":
        return [
          { label: "Mesneti kaldır", action: () => handleRemoveSupport(target.id) },
        ];
      case "point": {
        const rotationItems: ContextMenuItem[] = [
          {
            label: "Sağ",
            action: () => setPointAngle(target.id, 0),
            tooltip: "Açı değerini 0° yapar.",
          },
          {
            label: "Sol",
            action: () => setPointAngle(target.id, 180),
            tooltip: "Açı değerini 180° yapar.",
          },
          {
            label: "Alt",
            action: () => setPointAngle(target.id, -90),
            tooltip: "Açı değerini -90° yapar.",
          },
          {
            label: "Üst",
            action: () => setPointAngle(target.id, 90),
            tooltip: "Açı değerini 90° yapar.",
          },
          {
            label: "Sağa 45°",
            action: () => rotatePointAngle(target.id, 45),
            tooltip: "Açı değerini +45° döndürür.",
          },
          {
            label: "Sola 45°",
            action: () => rotatePointAngle(target.id, -45),
            tooltip: "Açı değerini -45° döndürür.",
          },
          {
            label: "Özel değer gir...",
            action: () => promptPointAngle(target.id),
            tooltip: "İstediğiniz derece değerini tanımlayın.",
          },
        ];
        return [
          {
            label: "Döndür",
            tooltip: "Yük yönünü ayarlayın veya döndürün.",
            submenu: rotationItems,
          },
          { label: "Tekil yükü kaldır", action: () => handleRemovePointLoad(target.id) },
        ];
      }
      case "udl": {
        const current = udls.find((item) => item.id === target.id);
        const shapeItems: ContextMenuItem[] = [
          {
            label: current?.shape === "uniform" ? "Düzgün (varsayılan)" : "Düzgün",
            disabled: current?.shape === "uniform",
            action: () => setUdlShape(target.id, "uniform"),
            tooltip: "Yoğunluk sabit kalır.",
          },
          {
            label: current?.shape === "triangular_increasing" ? "Üçgen: 0 → max" : "Üçgen (0 → max)",
            disabled: current?.shape === "triangular_increasing",
            action: () => setUdlShape(target.id, "triangular_increasing"),
            tooltip: "Başlangıçta sıfır, bitişte maksimum yoğunluk.",
          },
          {
            label: current?.shape === "triangular_decreasing" ? "Üçgen: max → 0" : "Üçgen (max → 0)",
            disabled: current?.shape === "triangular_decreasing",
            action: () => setUdlShape(target.id, "triangular_decreasing"),
            tooltip: "Başlangıçta maksimum, bitişte sıfır yoğunluk.",
          },
        ];

        return [
          {
            label: "Profil",
            tooltip: "Yayılı yükün dağılım biçimini seçin.",
            submenu: shapeItems,
          },
          { label: "Yönü değiştir", action: () => toggleUdlDirection(target.id) },
          { label: "Yayılı yükü kaldır", action: () => handleRemoveUdl(target.id) },
        ];
      }
      case "moment":
        return [
          { label: "Yönü değiştir", action: () => toggleMomentDirection(target.id) },
          { label: "Momenti kaldır", action: () => handleRemoveMoment(target.id) },
        ];
      default:
        return [];
    }
  }, [contextMenu, handleAddMoment, handleAddPointLoad, handleAddSupport, handleAddUdl, handleRemoveMoment, handleRemovePointLoad, handleRemoveSupport, handleRemoveUdl, promptPointAngle, rotatePointAngle, setPointAngle, setUdlShape, supports.length, toggleMomentDirection, toggleUdlDirection, udls]);

  return (
    <main
      className="pb-16"
      onClick={() => {
        if (contextMenu) {
          closeContextMenu();
        }
      }}
    >
      {/* Header */}
      <header className="border-b border-slate-800/50 bg-slate-900/80 backdrop-blur-md">
        <div className="mx-auto flex w-full max-w-none items-center justify-between px-4 py-3 sm:px-6">
          <div className="flex items-center gap-3">
            <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-white/95 shadow-lg ring-1 ring-slate-800/5">
              <Image src={KtoLogo} alt="KTO Logo" className="h-11 w-11 object-contain" priority />
            </div>
            <div>
              <h1 className="text-xl font-bold text-slate-50">Kiriş Moment Hesaplayıcı</h1>
              <p className="text-xs text-slate-400">Statik analiz aracı</p>
            </div>
          </div>
          <div className="hidden text-right sm:block">
            <p className="text-[10px] text-slate-500">Made by</p>
            <p className="text-xs font-semibold text-slate-300">Deha Özcan</p>
          </div>
        </div>
      </header>

      <div className="mx-auto flex w-full max-w-none flex-col gap-5 px-4 pt-4 sm:px-6">
        <section className="panel space-y-3 p-3 sm:p-4">
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
          <div className="flex flex-col gap-2 md:flex-row md:flex-wrap">
            {PRESETS.map((preset) => (
              <button
                key={preset.key}
                type="button"
                onClick={() => applyPreset(preset)}
                className={clsx(
                  "flex-1 rounded-xl border px-3 py-2 text-left transition",
                  activePreset === preset.key
                    ? "border-cyan-400 bg-cyan-500/10"
                    : "border-slate-700/80 bg-slate-900/60 hover:border-cyan-400",
                )}
              >
                <p className="text-xs font-semibold text-slate-100">{preset.title}</p>
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
              onPointLoadMagnitudeChange={handlePointLoadMagnitudeChange}
              onUdlMagnitudeChange={handleUdlMagnitudeChange}
              onMomentMagnitudeChange={handleMomentMagnitudeChange}
            />
            <BeamDiagrams
              x={diagramData.x}
              shear={diagramData.shear}
              moment={diagramData.moment}
              normal={diagramData.normal}
              loading={isPending}
            />
          </div>

          <div className="panel space-y-6 p-6">
            <div>
              <span className="tag">Çözüm</span>
              <p className="text-sm text-slate-400">Mesnet tepkileri ve denge kontrolü</p>
            </div>

            <div className="space-y-6">
              {result?.meta.recommendation && (
                <div className="panel-muted border border-cyan-500/30 bg-cyan-500/10 p-4 text-sm text-cyan-100">
                  <p className="text-[11px] font-semibold uppercase tracking-wide text-cyan-200/90">Tavsiye edilen yöntem</p>
                  <p className="mt-1 text-base font-semibold text-cyan-100">{result.meta.recommendation.title}</p>
                  <p className="mt-2 text-xs leading-relaxed text-cyan-100/90">{result.meta.recommendation.reason}</p>
                  <p className="mt-3 text-[11px] text-cyan-200/70">Mesnet reaksiyonlarından sonra bu yöntemle devam edin.</p>
                </div>
              )}

              {/* Detailed Solution Button */}
              {result?.detailed_solutions && (
                <button
                  onClick={() => setIsDetailedSolutionOpen(true)}
                  className="flex w-full items-center justify-center gap-2 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-500 px-4 py-3 text-sm font-semibold text-white shadow-lg transition hover:from-cyan-600 hover:to-blue-600 hover:shadow-xl"
                >
                  <svg
                    className="h-4 w-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    />
                  </svg>
                  Çözümü Göster
                </button>
              )}

              {/* Results Section */}
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-slate-300">Mesnet Tepkileri</h3>
                  {result?.meta.solve_time_ms !== undefined && (
                    <span className="rounded-full bg-slate-800/80 px-3 py-1 text-xs text-slate-300">
                      {result.meta.solve_time_ms.toFixed(2)} ms
                    </span>
                  )}
                </div>
                {error ? (
                  <div className="panel-muted border border-rose-500/40 bg-rose-500/10 p-4 text-sm text-rose-200">
                    {error}
                  </div>
                ) : (
                  <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                    {reactions && reactions.length > 0 ? (
                      reactions.map((reaction) => (
                        <div key={reaction.support_id} className="panel-muted p-4">
                          <p className="text-xs uppercase tracking-wide text-slate-400">
                            R<sub>{reaction.support_id}</sub> ({reaction.support_type})
                          </p>
                          <p className="text-3xl font-semibold text-cyan-300">
                            {reaction.vertical.toFixed(2)} kN
                          </p>
                          <p className="text-xs text-slate-500">x = {reaction.position.toFixed(2)} m</p>
                        </div>
                      ))
                    ) : null}
                    {reactions && reactions.length > 0 && (
                      <div className="panel-muted col-span-full p-4 text-sm text-slate-300">
                        <p className="font-medium text-slate-200">Denge kontrolü</p>
                        <p>Tepki toplamı = {reactions.reduce((sum, r) => sum + (r.vertical ?? 0), 0).toFixed(2)} kN</p>
                      </div>
                    )}
                  </div>
                )}

                {result?.meta.validation_warnings && result.meta.validation_warnings.length > 0 && (
                  <div className="panel-muted border border-amber-400/40 bg-amber-500/10 p-4 text-xs text-amber-100">
                    <p className="mb-2 font-semibold">Uyarılar</p>
                    <ul className="space-y-2">
                      {result.meta.validation_warnings.map((warning) => (
                        <li key={warning}>- {warning}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {contextMenu && (
        <div
          className="fixed z-50 min-w-[180px] rounded-xl border border-slate-700/80 bg-slate-900/95 p-2 text-sm text-slate-100 shadow-lg"
          style={{ top: contextMenu.clientY, left: contextMenu.clientX }}
          onClick={(event) => event.stopPropagation()}
        >
          {contextMenuItems.map((item) => {
            const hasSubmenu = Array.isArray(item.submenu) && item.submenu.length > 0;
            if (hasSubmenu) {
              const isOpen = openSubmenu === item.label;
              return (
                <div key={item.label} className="relative">
                  <button
                    type="button"
                    disabled={item.disabled}
                    onClick={() => {
                      if (item.disabled) {
                        return;
                      }
                      setOpenSubmenu(isOpen ? null : item.label);
                    }}
                    onMouseEnter={() => {
                      if (!item.disabled) {
                        setOpenSubmenu(item.label);
                      }
                    }}
                    title={item.tooltip}
                    className={clsx(
                      "flex w-full items-center gap-2 rounded-lg px-3 py-1.5 text-left transition",
                      item.disabled
                        ? "cursor-not-allowed text-slate-500"
                        : "hover:bg-slate-800 hover:text-white",
                    )}
                  >
                    <span>{item.label}</span>
                    <span className="ml-auto text-xs text-slate-500">›</span>
                  </button>
                  {isOpen && (
                    <div
                      className="absolute top-0 left-full ml-1 min-w-[180px] rounded-xl border border-slate-700/80 bg-slate-900/95 p-1 shadow-lg"
                      onClick={(event) => event.stopPropagation()}
                    >
                      {item.submenu!.map((subItem) => (
                        <button
                          key={subItem.label}
                          type="button"
                          disabled={subItem.disabled}
                          onClick={() => handleContextMenuSelection(subItem.action)}
                          title={subItem.tooltip}
                          className={clsx(
                            "flex w-full items-center gap-2 rounded-lg px-3 py-1.5 text-left transition",
                            subItem.disabled
                              ? "cursor-not-allowed text-slate-500"
                              : "hover:bg-slate-800 hover:text-white",
                          )}
                        >
                          {subItem.label}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              );
            }
            return (
              <button
                key={item.label}
                type="button"
                disabled={item.disabled}
                onClick={() => handleContextMenuSelection(item.action)}
                title={item.tooltip}
                className={clsx(
                  "flex w-full items-center gap-2 rounded-lg px-3 py-1.5 text-left transition",
                  item.disabled
                    ? "cursor-not-allowed text-slate-500"
                    : "hover:bg-slate-800 hover:text-white",
                )}
              >
                {item.label}
              </button>
            );
          })}
          {contextMenuItems.length === 0 && (
            <div className="px-3 py-1.5 text-xs text-slate-500">No actions available</div>
          )}
        </div>
      )}

      {/* Detailed Solution Panel */}
      {result?.detailed_solutions && (
        <DetailedSolutionPanel
          detailedSolution={result.detailed_solutions}
          isOpen={isDetailedSolutionOpen}
          onClose={() => setIsDetailedSolutionOpen(false)}
        />
      )}
    </main>
  );
}
