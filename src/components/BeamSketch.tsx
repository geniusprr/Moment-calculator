"use client";

import { CSSProperties, useCallback, useEffect, useMemo, useRef, useState } from "react";

import type {
  MomentLoadInput,
  PointLoadInput,
  SupportInput,
  SupportReaction,
  UdlInput,
  LoadColorConfig,
} from "@/types/beam";

export type SketchContextTarget =
  | { kind: "blank"; x: number }
  | { kind: "support"; id: string; x: number }
  | { kind: "point"; id: string; x: number }
  | { kind: "udl"; id: string; x: number }
  | { kind: "moment"; id: string; x: number };

interface BeamSketchProps {
  length: number;
  supports: SupportInput[];
  pointLoads: PointLoadInput[];
  udls: UdlInput[];
  momentLoads: MomentLoadInput[];
  reactions?: SupportReaction[];
  loadColors?: Partial<LoadColorConfig>;
  onSupportPositionChange: (id: string, position: number) => void;
  onPointLoadPositionChange: (id: string, position: number) => void;
  onUdlRangeChange: (id: string, edge: "start" | "end", position: number) => void;
  onMomentPositionChange: (id: string, position: number) => void;
  onOpenContextMenu: (target: SketchContextTarget, clientX: number, clientY: number) => void;
  onPointLoadMagnitudeChange?: (id: string, magnitude: number) => void;
  onUdlMagnitudeChange?: (id: string, magnitude: number) => void;
  onMomentMagnitudeChange?: (id: string, magnitude: number) => void;
}

interface DragState {
  type: "support" | "point" | "moment" | "udl-start" | "udl-end" | "udl-center";
  id: string;
  pointerId: number;
  offset: number;
}

type DragTarget = {
  type: DragState["type"];
  id: string;
};

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

const DEFAULT_LOAD_COLORS: LoadColorConfig = {
  point: "#ef4444",
  uniformUdl: "#f97316",
  triangularUdl: "#a855f7",
  moment: "#22c55e",
};

const parseHex = (color: string): [number, number, number] | null => {
  const value = color.trim();
  if (!value.startsWith("#")) {
    return null;
  }
  const hex = value.slice(1);
  if (hex.length === 3) {
    const r = parseInt(hex[0] + hex[0], 16);
    const g = parseInt(hex[1] + hex[1], 16);
    const b = parseInt(hex[2] + hex[2], 16);
    return [r, g, b];
  }
  if (hex.length === 6) {
    const r = parseInt(hex.slice(0, 2), 16);
    const g = parseInt(hex.slice(2, 4), 16);
    const b = parseInt(hex.slice(4, 6), 16);
    return [r, g, b];
  }
  return null;
};

const applyAlpha = (color: string, alpha: number): string => {
  const rgb = parseHex(color);
  if (!rgb) {
    return color;
  }
  const [r, g, b] = rgb;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
};

const mixColors = (color: string, mixWith: string, amount: number): string => {
  const base = parseHex(color);
  const mix = parseHex(mixWith);
  if (!base || !mix) {
    return color;
  }
  const clampAmount = clamp(amount, 0, 1);
  const r = Math.round(base[0] * (1 - clampAmount) + mix[0] * clampAmount);
  const g = Math.round(base[1] * (1 - clampAmount) + mix[1] * clampAmount);
  const b = Math.round(base[2] * (1 - clampAmount) + mix[2] * clampAmount);
  return `#${[r, g, b]
    .map((component) => component.toString(16).padStart(2, "0"))
    .join("")}`;
};

const getReadableTextColor = (color: string): string => {
  const rgb = parseHex(color);
  if (!rgb) {
    return "#f8fafc";
  }
  const toLinear = (value: number) => {
    const channel = value / 255;
    return channel <= 0.03928 ? channel / 12.92 : Math.pow((channel + 0.055) / 1.055, 2.4);
  };
  const [r, g, b] = rgb.map(toLinear) as [number, number, number];
  const luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  return luminance > 0.55 ? "#0f172a" : "#f8fafc";
};

export function BeamSketch({
  length,
  supports,
  pointLoads,
  udls,
  momentLoads,
  reactions,
  loadColors,
  onSupportPositionChange,
  onPointLoadPositionChange,
  onUdlRangeChange,
  onMomentPositionChange,
  onOpenContextMenu,
  onPointLoadMagnitudeChange,
  onUdlMagnitudeChange,
  onMomentMagnitudeChange,
}: BeamSketchProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [dragState, setDragState] = useState<DragState | null>(null);
  const [editingValue, setEditingValue] = useState<{
    type: "pointLoad" | "udl" | "moment" | "dimension";
    id: string;
    value: string;
  } | null>(null);

  const beamLength = Math.max(length, 1e-4);

  const colors = useMemo(() => ({ ...DEFAULT_LOAD_COLORS, ...(loadColors ?? {}) }), [loadColors]);
  const pointColor = colors.point;
  const uniformUdlColor = colors.uniformUdl;
  const triangularUdlColor = colors.triangularUdl;
  const momentColor = colors.moment;
  const momentCwColor = useMemo(() => mixColors(momentColor, "#1e293b", 0.35), [momentColor]);

  const reactionMap = useMemo(() => {
    if (!reactions) {
      return new Map<string, SupportReaction>();
    }
    return new Map(reactions.map((reaction) => [reaction.support_id.toUpperCase(), reaction]));
  }, [reactions]);

  const positionToPercent = useCallback(
    (position: number) => clamp(position / beamLength, 0, 1) * 100,
    [beamLength],
  );

  const valueFromPointer = useCallback(
    (clientX: number) => {
      const container = containerRef.current;
      if (!container) {
        return 0;
      }
      const rect = container.getBoundingClientRect();
      if (rect.width <= 0) {
        return 0;
      }
      const ratio = clamp((clientX - rect.left) / rect.width, 0, 1);
      return ratio * beamLength;
    },
    [beamLength],
  );

  const applyPosition = useCallback(
    (target: DragState | null, pointerValue: number) => {
      if (!target) {
        return;
      }

      const snap = (value: number) => Math.round(value * 100) / 100;
      const desired = pointerValue - target.offset;

      switch (target.type) {
        case "support": {
          const safe = clamp(desired, 0, beamLength);
          onSupportPositionChange(target.id, snap(safe));
          break;
        }
        case "point": {
          const safe = clamp(desired, 0, beamLength);
          onPointLoadPositionChange(target.id, snap(safe));
          break;
        }
        case "moment": {
          const safe = clamp(desired, 0, beamLength);
          onMomentPositionChange(target.id, snap(safe));
          break;
        }
        case "udl-start": {
          const udl = udls.find((item) => item.id === target.id);
          if (!udl) {
            return;
          }
          const safe = clamp(desired, 0, beamLength);
          const maxStart = udl.end - 0.1;
          const constrained = Math.min(safe, maxStart);
          const snapped = Math.min(snap(constrained), maxStart);
          onUdlRangeChange(target.id, "start", clamp(snapped, 0, beamLength - 0.1));
          break;
        }
        case "udl-end": {
          const udl = udls.find((item) => item.id === target.id);
          if (!udl) {
            return;
          }
          const safe = clamp(desired, 0, beamLength);
          const minEnd = udl.start + 0.1;
          const constrained = Math.max(safe, minEnd);
          const snapped = Math.max(snap(constrained), minEnd);
          onUdlRangeChange(target.id, "end", clamp(snapped, minEnd, beamLength));
          break;
        }
        case "udl-center": {
          const udl = udls.find((item) => item.id === target.id);
          if (!udl) {
            return;
          }
          const span = Math.max(udl.end - udl.start, 0.1);
          const half = span / 2;
          const rawMid = clamp(desired, half, beamLength - half);
          const snappedMid = clamp(snap(rawMid), half, beamLength - half);
          let nextStart = snappedMid - half;
          let nextEnd = snappedMid + half;

          if (nextStart < 0) {
            nextStart = 0;
            nextEnd = span;
          }
          if (nextEnd > beamLength) {
            nextEnd = beamLength;
            nextStart = beamLength - span;
          }

          let snappedStart = snap(nextStart);
          let snappedEnd = snap(nextEnd);
          const desiredSpan = Math.max(0.1, snap(span));

          if (snappedEnd - snappedStart < 0.1) {
            snappedStart = clamp(snappedMid - desiredSpan / 2, 0, beamLength - desiredSpan);
            snappedStart = snap(snappedStart);
            snappedEnd = snap(Math.min(snappedStart + desiredSpan, beamLength));
          }

          if (snappedEnd - snappedStart < 0.1) {
            snappedEnd = snap(Math.min(beamLength, snappedStart + 0.1));
          }

          snappedStart = clamp(snappedStart, 0, beamLength - 0.1);
          snappedEnd = clamp(snappedEnd, snappedStart + 0.1, beamLength);

          onUdlRangeChange(target.id, "start", snappedStart);
          onUdlRangeChange(target.id, "end", snappedEnd);
          break;
        }
      }
    },
    [beamLength, onMomentPositionChange, onPointLoadPositionChange, onSupportPositionChange, onUdlRangeChange, udls],
  );

  useEffect(() => {
    if (!dragState) {
      return;
    }

    const handlePointerMove = (event: PointerEvent) => {
      if (event.pointerId !== dragState.pointerId) {
        return;
      }
      const value = valueFromPointer(event.clientX);
      applyPosition(dragState, value);
    };

    const handlePointerUp = (event: PointerEvent) => {
      if (event.pointerId !== dragState.pointerId) {
        return;
      }
      setDragState(null);
    };

    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);
    window.addEventListener("pointercancel", handlePointerUp);

    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
      window.removeEventListener("pointercancel", handlePointerUp);
    };
  }, [applyPosition, dragState, valueFromPointer]);

  const resolveCurrentPosition = useCallback(
    (target: DragTarget): number => {
      switch (target.type) {
        case "support": {
          const support = supports.find((item) => item.id === target.id);
          return support ? support.position : 0;
        }
        case "point": {
          const load = pointLoads.find((item) => item.id === target.id);
          return load ? load.position : 0;
        }
        case "moment": {
          const moment = momentLoads.find((item) => item.id === target.id);
          return moment ? moment.position : 0;
        }
        case "udl-start": {
          const udl = udls.find((item) => item.id === target.id);
          return udl ? udl.start : 0;
        }
        case "udl-end": {
          const udl = udls.find((item) => item.id === target.id);
          return udl ? udl.end : 0;
        }
        case "udl-center": {
          const udl = udls.find((item) => item.id === target.id);
          if (!udl) {
            return 0;
          }
          return (udl.start + udl.end) / 2;
        }
        default:
          return 0;
      }
    },
    [momentLoads, pointLoads, supports, udls],
  );

  const beginDrag = useCallback(
    (target: DragTarget) => (event: React.PointerEvent<HTMLDivElement | HTMLButtonElement>) => {
      event.preventDefault();
      event.stopPropagation();

      const pointerValue = valueFromPointer(event.clientX);
      const currentValue = resolveCurrentPosition(target);
      const offset = pointerValue - currentValue;
      const state: DragState = { ...target, pointerId: event.pointerId, offset };

      setDragState(state);
      applyPosition(state, pointerValue);
    },
    [applyPosition, resolveCurrentPosition, valueFromPointer],
  );

  const openContextMenu = useCallback(
    (event: React.MouseEvent, target: SketchContextTarget) => {
      event.preventDefault();
      event.stopPropagation();
      onOpenContextMenu(target, event.clientX, event.clientY);
    },
    [onOpenContextMenu],
  );

  const handleValueClick = useCallback(
    (event: React.MouseEvent, type: "pointLoad" | "udl" | "moment", id: string, currentValue: number) => {
      event.preventDefault();
      event.stopPropagation();
      setEditingValue({ type, id, value: currentValue.toFixed(1) });
    },
    [],
  );

  const handleValueInputChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    if (!editingValue) return;
    setEditingValue({ ...editingValue, value: event.target.value });
  }, [editingValue]);

  const handleValueInputBlur = useCallback(() => {
    if (!editingValue) return;

    const numValue = parseFloat(editingValue.value);
    if (!isNaN(numValue) && numValue > 0) {
      switch (editingValue.type) {
        case "pointLoad":
          onPointLoadMagnitudeChange?.(editingValue.id, numValue);
          break;
        case "udl":
          onUdlMagnitudeChange?.(editingValue.id, numValue);
          break;
        case "moment":
          onMomentMagnitudeChange?.(editingValue.id, numValue);
          break;
      }
    }
    setEditingValue(null);
  }, [editingValue, onPointLoadMagnitudeChange, onUdlMagnitudeChange, onMomentMagnitudeChange]);

  const handleValueInputKeyDown = useCallback((event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter") {
      event.currentTarget.blur();
    } else if (event.key === "Escape") {
      setEditingValue(null);
    }
  }, []);

  const supportMarkers = useMemo(
    () =>
      supports.map((support) => ({
        ...support,
        percent: positionToPercent(support.position),
        reaction: reactionMap.get(support.id.toUpperCase()),
      })),
    [supports, positionToPercent, reactionMap],
  );

  const pointHandles = useMemo(
    () =>
      pointLoads.map((load) => ({
        ...load,
        percent: positionToPercent(load.position),
      })),
    [pointLoads, positionToPercent],
  );

  const udlSpans = useMemo(
    () =>
      udls.map((load) => ({
        ...load,
        startPercent: positionToPercent(load.start),
        endPercent: positionToPercent(load.end),
      })),
    [udls, positionToPercent],
  );

  const momentHandles = useMemo(
    () =>
      momentLoads.map((moment) => ({
        ...moment,
        percent: positionToPercent(moment.position),
      })),
    [momentLoads, positionToPercent],
  );

  const dimensionSegments = useMemo(() => {
    const stops = new Set<number>([0, length]);
    supports.forEach((support) => stops.add(clamp(support.position, 0, length)));
    pointLoads.forEach((load) => stops.add(clamp(load.position, 0, length)));
    udls.forEach((load) => {
      stops.add(clamp(load.start, 0, length));
      stops.add(clamp(load.end, 0, length));
    });
    momentLoads.forEach((moment) => stops.add(clamp(moment.position, 0, length)));
    const sorted = Array.from(stops).sort((a, b) => a - b);
    const segments: Array<{ start: number; end: number }> = [];
    for (let index = 0; index < sorted.length - 1; index += 1) {
      const start = sorted[index];
      const end = sorted[index + 1];
      if (end - start > 1e-6) {
        segments.push({ start, end });
      }
    }
    return segments;
  }, [length, momentLoads, pointLoads, supports, udls]);

  const UDL_AREA_HEIGHT = 64;
  const UDL_TOP = `calc(50% - ${12 + UDL_AREA_HEIGHT}px)`;
  const UDL_ARROW_HEAD_HEIGHT = 12;
  const UDL_ARROW_BODY_HEIGHT = UDL_AREA_HEIGHT - UDL_ARROW_HEAD_HEIGHT;
  const UDL_ARROW_WIDTH = 18;
  const UDL_ARROW_HALF_WIDTH = UDL_ARROW_WIDTH / 2;

  return (
    <div
      className="panel space-y-3 p-4"
      onContextMenu={(event) => {
        const x = valueFromPointer(event.clientX);
        openContextMenu(event, { kind: "blank", x });
      }}
    >
      <div ref={containerRef} className="relative h-64 select-none">
        {/* Realistic Steel Beam (I-Profile) */}
        <div
          className="absolute left-[2%] right-[2%] h-6 flex flex-col shadow-md"
          style={{ top: 'calc(50% - 12px)' }}
        >
          <div className="h-1 w-full bg-slate-500 rounded-t-sm" /> {/* Top flange */}
          <div className="flex-1 w-full bg-gradient-to-b from-slate-300 via-slate-200 to-slate-300 border-x border-slate-300/50" /> {/* Web */}
          <div className="h-1 w-full bg-slate-500 rounded-b-sm" /> {/* Bottom flange */}
        </div>

        {/* Simplified support markers - below the beam, tip touching it */}
        {supportMarkers.map((support) => {
          const isRight = support.percent > 50;
          return (
            <div
              key={support.id}
              className="absolute flex translate-x-[-50%] flex-col items-center gap-1"
              style={{
                left: support.type === "fixed"
                  ? (isRight ? `calc(${2 + (support.percent * 0.96)}% + 8px)` : `calc(${2 + (support.percent * 0.96)}% - 8px)`)
                  : `${2 + (support.percent * 0.96)}%`,
                top: support.type === "fixed" ? 'calc(50% - 70px)' : 'calc(50% + 12px)'
              }}
              onContextMenu={(event) => openContextMenu(event, { kind: "support", id: support.id, x: support.position })}
            >
              {support.type === "fixed" && <span className="text-xs font-semibold text-cyan-200 mb-1">{support.id}</span>}

              <div
                className="cursor-ew-resize"
                onPointerDown={beginDrag({ type: "support", id: support.id })}
                title={support.reaction ? `Düşey=${support.reaction.vertical.toFixed(2)} kN${Math.abs(support.reaction.axial) > 1e-3 ? `, Yatay=${support.reaction.axial.toFixed(2)} kN` : ""}` : ""}
              >
                {support.type === "fixed" ? (
                  <div className="relative flex items-center justify-center">
                    {/* Wall - Concrete look */}
                    <div className="h-24 w-4 rounded-sm bg-[#9ca3af] border-2 border-[#4b5563] shadow-xl" />
                    {/* Hatching hints */}
                    <div
                      className={`absolute top-0 bottom-0 flex w-4 flex-col justify-between py-2 ${isRight ? "-right-4" : "-left-4"}`}
                    >
                      {[...Array(10)].map((_, i) => (
                        <div
                          key={i}
                          className={`h-0.5 w-full bg-[#6b7280] ${isRight ? "rotate-[45deg]" : "rotate-[-45deg]"}`}
                        />
                      ))}
                    </div>
                  </div>
                ) : (
                  <>
                    {/* Larger support symbol */}
                    <div className="h-0 w-0 border-x-[14px] border-b-[22px] border-x-transparent border-b-cyan-400" />
                    <div className="mx-auto h-1.5 w-7 rounded-b bg-slate-600" />
                  </>
                )}
              </div>
              {support.type !== "fixed" && <span className="text-xs font-semibold text-cyan-200">{support.id}</span>}
            </div>
          );
        })}

        {/* Simplified point loads - above the beam, touching it */}
        {pointHandles.map((load) => (
          <div
            key={load.id}
            className="absolute flex translate-x-[-50%] flex-col items-center cursor-ew-resize z-40"
            style={{ left: `${2 + (load.percent * 0.96)}%`, bottom: 'calc(50% + 12px)' }}
            onContextMenu={(event) => openContextMenu(event, { kind: "point", id: load.id, x: load.position })}
            onPointerDown={beginDrag({ type: "point", id: load.id })}
          >
            {editingValue?.type === "pointLoad" && editingValue.id === load.id ? (
              <input
                type="number"
                autoFocus
                className="mb-0.5 w-20 rounded px-1.5 py-0.5 text-xs font-semibold text-center outline-none ring-2 ring-white"
                value={editingValue.value}
                onChange={handleValueInputChange}
                onBlur={handleValueInputBlur}
                onKeyDown={handleValueInputKeyDown}
                onClick={(e) => e.stopPropagation()}
                onPointerDown={(e) => e.stopPropagation()}
                style={{
                  backgroundColor: applyAlpha(pointColor, 0.92),
                  color: getReadableTextColor(pointColor),
                }}
              />
            ) : (
              <span
                className="mb-0.5 rounded px-1.5 py-0.5 text-xs font-semibold cursor-pointer hover:ring-2 hover:ring-white/60 transition-all pointer-events-auto"
                style={{
                  backgroundColor: applyAlpha(pointColor, 0.9),
                  color: getReadableTextColor(pointColor),
                }}
                onClick={(e) => handleValueClick(e, "pointLoad", load.id, load.magnitude)}
                title="Değeri düzenlemek için tıklayın"
              >
                {load.id}: {load.magnitude.toFixed(1)}kN
              </span>
            )}
            {/* Arrow pointing down towards the beam */}
            <svg
              className="h-16 w-6"
              viewBox="0 0 24 64"
              style={{ transform: `rotate(${-(load.angleDeg + 90)}deg)` }}
            >
              <title>{`${load.id} açısı: ${load.angleDeg}°`}</title>
              <line x1="12" y1="0" x2="12" y2="54" stroke={pointColor} strokeWidth="2.5" />
              <polygon points="6,54 18,54 12,64" fill={pointColor} />
            </svg>
          </div>
        ))}

        {/* UDL spans with improved arrow alignment */}
        {udlSpans.map((load) => {
          const startPercent = 2 + load.startPercent * 0.96;
          const widthPercent = (load.endPercent - load.startPercent) * 0.96;
          const arrowCount = Math.max(5, Math.min(11, Math.round(widthPercent / 5)));
          const labelText = load.shape === "uniform"
            ? `${load.id}: ${load.magnitude.toFixed(1)} kN/m`
            : `${load.id}: max ${load.magnitude.toFixed(1)} kN/m`;

          const isUpwardLoad = load.direction === "up";
          const baseLinePosition: CSSProperties = isUpwardLoad ? { bottom: 0 } : { top: 0 };
          const getScaleForRatio = (ratio: number) => {
            if (load.shape === "uniform") {
              return 1;
            }
            const triangularRatio = load.shape === "triangular_increasing" ? ratio : 1 - ratio;
            // Use full range 0 to 1 for triangular loads (no minimum scale)
            return triangularRatio;
          };
          const startScale = getScaleForRatio(0);
          const endScale = getScaleForRatio(1);
          const baseColor = load.shape === "uniform" ? uniformUdlColor : triangularUdlColor;
          const labelTextColor = getReadableTextColor(baseColor);
          const labelBackground = applyAlpha(baseColor, 0.88);

          // Calculate line positions to touch top of scaled arrows (body + head both scaled)
          const startArrowTotalHeight = (UDL_ARROW_BODY_HEIGHT + UDL_ARROW_HEAD_HEIGHT) * startScale;
          const endArrowTotalHeight = (UDL_ARROW_BODY_HEIGHT + UDL_ARROW_HEAD_HEIGHT) * endScale;

          const arrowPositions = arrowCount > 1
            ? Array.from({ length: arrowCount }, (_, index) => (index / (arrowCount - 1)) * 100)
            : [50];

          return (
            <div
              key={load.id}
              className="absolute z-30"
              style={{
                left: `${startPercent}%`,
                width: `${widthPercent}%`,
                top: UDL_TOP,
                height: `${UDL_AREA_HEIGHT}px`,
              }}
              onContextMenu={(event) =>
                openContextMenu(event, { kind: "udl", id: load.id, x: (load.start + load.end) / 2 })
              }
            >
              <div
                className="absolute pointer-events-auto"
                style={{
                  top: -32,
                  left: "50%",
                  transform: "translateX(-50%)",
                  whiteSpace: "nowrap",
                  maxWidth: "100%",
                }}
              >
                {editingValue?.type === "udl" && editingValue.id === load.id ? (
                  <input
                    type="number"
                    autoFocus
                    className="w-20 rounded px-2 py-0.5 text-xs font-semibold text-center outline-none ring-2 ring-white"
                    value={editingValue.value}
                    onChange={handleValueInputChange}
                    onBlur={handleValueInputBlur}
                    onKeyDown={handleValueInputKeyDown}
                    onClick={(e) => e.stopPropagation()}
                    onPointerDown={(e) => e.stopPropagation()}
                    style={{
                      backgroundColor: applyAlpha(baseColor, 0.92),
                      color: labelTextColor,
                    }}
                  />
                ) : (
                  <span
                    className="rounded px-2 py-0.5 text-xs font-semibold cursor-pointer hover:ring-2 hover:ring-white/60 transition-all"
                    style={{
                      backgroundColor: labelBackground,
                      color: labelTextColor,
                    }}
                    onClick={(e) => handleValueClick(e, "udl", load.id, load.magnitude)}
                    title="Değeri düzenlemek için tıklayın"
                  >
                    {labelText}
                  </span>
                )}
              </div>

              {load.shape === "uniform" ? (
                <div
                  className="absolute left-0 right-0 h-1"
                  style={{ ...baseLinePosition, backgroundColor: uniformUdlColor }}
                />
              ) : (
                <svg
                  className="absolute pointer-events-none"
                  style={{
                    bottom: 0,
                    left: 0,
                    width: `100%`,
                    height: `${UDL_AREA_HEIGHT}px`
                  }}
                  viewBox={`0 0 100 ${UDL_AREA_HEIGHT}`}
                  preserveAspectRatio="none"
                >
                  <line
                    x1="0"
                    y1={UDL_AREA_HEIGHT - startArrowTotalHeight}
                    x2="100"
                    y2={UDL_AREA_HEIGHT - endArrowTotalHeight}
                    stroke={triangularUdlColor}
                    strokeWidth="2.5"
                    vectorEffect="non-scaling-stroke"
                  />
                </svg>
              )}

              <div
                className="absolute inset-0 cursor-grab transition hover:opacity-90"
                onPointerDown={beginDrag({ type: "udl-center", id: load.id })}
                title="Yayılı yükü sürükle"
              >
                {arrowPositions.map((position, index) => {
                  const ratio = arrowCount > 1 ? index / (arrowCount - 1) : 0.5;
                  const arrowScale = getScaleForRatio(ratio);
                  // Scale both body and head proportionally
                  const scaledBodyHeight = UDL_ARROW_BODY_HEIGHT * arrowScale;
                  const scaledHeadHeight = UDL_ARROW_HEAD_HEIGHT * arrowScale;
                  const totalHeight = scaledBodyHeight + scaledHeadHeight;
                  const scaledHeadWidth = 5 * arrowScale;
                  const transforms = ["translateX(-50%)"];
                  if (load.direction === "up") {
                    transforms.push("rotate(180deg)");
                  }
                  return (
                    <svg
                      key={`${load.id}-arrow-${index}`}
                      className="absolute"
                      style={{ left: `${position}%`, transform: transforms.join(" "), bottom: 0 }}
                      width={UDL_ARROW_WIDTH}
                      height={UDL_AREA_HEIGHT}
                      viewBox={`0 0 ${UDL_ARROW_WIDTH} ${UDL_AREA_HEIGHT}`}
                    >
                      <line
                        x1={UDL_ARROW_HALF_WIDTH}
                        y1={UDL_AREA_HEIGHT - totalHeight + 1}
                        x2={UDL_ARROW_HALF_WIDTH}
                        y2={UDL_AREA_HEIGHT - scaledHeadHeight}
                        stroke={baseColor}
                        strokeWidth="2"
                      />
                      <polygon
                        points={`${UDL_ARROW_HALF_WIDTH - scaledHeadWidth},${UDL_AREA_HEIGHT - scaledHeadHeight} ${UDL_ARROW_HALF_WIDTH + scaledHeadWidth},${UDL_AREA_HEIGHT - scaledHeadHeight} ${UDL_ARROW_HALF_WIDTH},${UDL_AREA_HEIGHT}`}
                        fill={baseColor}
                      />
                    </svg>
                  );
                })}
              </div>

              <div
                className="absolute z-40 w-8 translate-x-[-50%] cursor-ew-resize"
                style={{ left: "0%", top: "8px", height: `${UDL_AREA_HEIGHT - 16}px` }}
                onPointerDown={beginDrag({ type: "udl-start", id: load.id })}
                title="Yayılı yük başlangıcını sürükle"
              />

              <div
                className="absolute z-40 w-8 translate-x-[-50%] cursor-ew-resize"
                style={{ left: "100%", top: "8px", height: `${UDL_AREA_HEIGHT - 16}px` }}
                onPointerDown={beginDrag({ type: "udl-end", id: load.id })}
                title="Yayılı yük bitişini sürükle"
              />
            </div>
          );
        })}

        {/* Simplified moment loads - above the beam, touching it */}
        {momentHandles.map((moment) => (
          <div
            key={moment.id}
            className="absolute flex translate-x-[-50%] flex-col items-center gap-1 cursor-ew-resize z-40"
            style={{ left: `${2 + (moment.percent * 0.96)}%`, bottom: 'calc(50% + 14px)' }}
            onContextMenu={(event) => openContextMenu(event, { kind: "moment", id: moment.id, x: moment.position })}
            onPointerDown={beginDrag({ type: "moment", id: moment.id })}
          >
            {editingValue?.type === "moment" && editingValue.id === moment.id ? (
              <input
                type="number"
                autoFocus
                className="w-20 rounded px-2 py-0.5 text-xs font-semibold text-center outline-none ring-2 ring-white"
                value={editingValue.value}
                onChange={handleValueInputChange}
                onBlur={handleValueInputBlur}
                onKeyDown={handleValueInputKeyDown}
                onClick={(e) => e.stopPropagation()}
                onPointerDown={(e) => e.stopPropagation()}
                style={{
                  backgroundColor: applyAlpha(momentColor, 0.8),
                  color: getReadableTextColor(momentColor),
                }}
              />
            ) : (
              <span
                className="rounded px-2 py-0.5 text-xs font-semibold pointer-events-auto cursor-pointer hover:ring-2 hover:ring-white/60 transition-all"
                style={{
                  backgroundColor: applyAlpha(momentColor, 0.8),
                  color: getReadableTextColor(momentColor),
                }}
                onClick={(e) => handleValueClick(e, "moment", moment.id, moment.magnitude)}
                title="Değeri düzenlemek için tıklayın"
              >
                {moment.magnitude.toFixed(1)} kN·m
              </span>
            )}
            <div
              className="flex h-8 w-8 items-center justify-center rounded-full border-2 text-lg font-bold"
              style={{
                borderColor: moment.direction === "ccw" ? momentColor : momentCwColor,
                color: moment.direction === "ccw" ? momentColor : momentCwColor,
                backgroundColor: moment.direction === "ccw"
                  ? applyAlpha(momentColor, 0.15)
                  : applyAlpha(momentCwColor, 0.15),
              }}
            >
              {moment.direction === "ccw" ? "↻" : "↺"}
            </div>
          </div>
        ))}

        {/* Basit uç etiketleri (biraz daha büyük) */}
        <div className="absolute bottom-2 left-[2%] right-[2%] flex justify-between text-sm text-slate-500">
          <span>0 m</span>
          <span>{beamLength.toFixed(1)} m</span>
        </div>

        {/* Dinamik parça mesafeleri (komşu noktalar arası) - below the beam with dimension lines */}
        {dimensionSegments.map((seg, idx) => {
          const mid = (seg.start + seg.end) / 2;
          const midPercent = positionToPercent(mid);
          const startPercent = positionToPercent(seg.start);
          const endPercent = positionToPercent(seg.end);
          const text = `${(seg.end - seg.start).toFixed(2)} m`;
          return (
            <div key={`dim-${idx}`}>
              {/* Start vertical line */}
              <div
                className="absolute w-0.5 h-8 bg-slate-400"
                style={{ left: `${2 + startPercent * 0.96}%`, top: "calc(50% + 60px)" }}
              />
              {/* End vertical line */}
              <div
                className="absolute w-0.5 h-8 bg-slate-400"
                style={{ left: `${2 + endPercent * 0.96}%`, top: "calc(50% + 60px)" }}
              />
              {/* Horizontal dimension line */}
              <div
                className="absolute h-0.5 bg-slate-400"
                style={{
                  left: `${2 + startPercent * 0.96}%`,
                  width: `${(endPercent - startPercent) * 0.96}%`,
                  top: "calc(50% + 68px)"
                }}
              />
              {/* Distance text */}
              <span
                className="absolute text-xs font-medium text-slate-300 bg-slate-800/80 px-1.5 py-0.5 rounded"
                style={{ left: `${2 + midPercent * 0.96}%`, transform: "translateX(-50%)", top: "calc(50% + 73px)" }}
              >
                {text}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
