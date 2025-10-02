"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import clsx from "clsx";

import type {
  MomentLoadInput,
  PointLoadInput,
  SupportInput,
  SupportReaction,
  UdlInput,
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
  onSupportPositionChange: (id: string, position: number) => void;
  onPointLoadPositionChange: (id: string, position: number) => void;
  onUdlRangeChange: (id: string, edge: "start" | "end", position: number) => void;
  onMomentPositionChange: (id: string, position: number) => void;
  onOpenContextMenu: (target: SketchContextTarget, clientX: number, clientY: number) => void;
}

interface DragState {
  type: "support" | "point" | "moment" | "udl-start" | "udl-end" | "udl-center";
  id: string;
  pointerId: number;
}

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

export function BeamSketch({
  length,
  supports,
  pointLoads,
  udls,
  momentLoads,
  reactions,
  onSupportPositionChange,
  onPointLoadPositionChange,
  onUdlRangeChange,
  onMomentPositionChange,
  onOpenContextMenu,
}: BeamSketchProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [dragState, setDragState] = useState<DragState | null>(null);

  const beamLength = Math.max(length, 1e-4);

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
    (target: DragState | null, value: number) => {
      if (!target) {
        return;
      }
      // Ensure supports cannot exceed beam length (0 to beamLength)
      const safe = clamp(value, 0, beamLength);
      switch (target.type) {
        case "support":
          onSupportPositionChange(target.id, safe);
          break;
        case "point":
          onPointLoadPositionChange(target.id, safe);
          break;
        case "moment":
          onMomentPositionChange(target.id, safe);
          break;
        case "udl-start": {
          const udl = udls.find((item) => item.id === target.id);
          if (!udl) {
            return;
          }
          const nextStart = Math.min(safe, udl.end - 0.1);
          onUdlRangeChange(target.id, "start", clamp(nextStart, 0, beamLength));
          break;
        }
        case "udl-end": {
          const udl = udls.find((item) => item.id === target.id);
          if (!udl) {
            return;
          }
          const nextEnd = Math.max(safe, udl.start + 0.1);
          onUdlRangeChange(target.id, "end", clamp(nextEnd, 0, beamLength));
          break;
        }
        case "udl-center": {
          const udl = udls.find((item) => item.id === target.id);
          if (!udl) {
            return;
          }
          // Move entire UDL span keeping its length
          const span = Math.max(udl.end - udl.start, 0.1);
          const half = span / 2;
          const mid = clamp(safe, half, beamLength - half);
          const nextStart = clamp(mid - half, 0, beamLength - 0.1);
          const nextEnd = clamp(mid + half, nextStart + 0.1, beamLength);
          onUdlRangeChange(target.id, "start", nextStart);
          onUdlRangeChange(target.id, "end", nextEnd);
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

  const beginDrag = useCallback(
    (target: DragState) => (event: React.PointerEvent<HTMLDivElement | HTMLButtonElement>) => {
      event.preventDefault();
      event.stopPropagation();
      const value = valueFromPointer(event.clientX);
      applyPosition(target, value);
      setDragState({ ...target, pointerId: event.pointerId });
    },
    [applyPosition, valueFromPointer],
  );

  const openContextMenu = useCallback(
    (event: React.MouseEvent, target: SketchContextTarget) => {
      event.preventDefault();
      event.stopPropagation();
      onOpenContextMenu(target, event.clientX, event.clientY);
    },
    [onOpenContextMenu],
  );

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

  return (
    <div
      className="panel space-y-3 p-4"
      onContextMenu={(event) => {
        const x = valueFromPointer(event.clientX);
        openContextMenu(event, { kind: "blank", x });
      }}
    >
      <div className="flex items-center justify-between">
        <span className="tag">Kiriş Çizimi</span>
        <p className="text-xs text-slate-400">Sürükleyerek ayarlayın. Hızlı işlemler için sağ tıklayın.</p>
      </div>
      <div ref={containerRef} className="relative h-40 select-none rounded-xl border border-slate-800/60 bg-slate-900/60">
        {/* Simplified beam bar - reduced side margins to fill width */}
        <div className="absolute left-[2%] right-[2%] top-1/2 h-3 -translate-y-1/2 rounded bg-slate-200/80 shadow-sm" />

        {/* Simplified support markers (larger) */}
        {supportMarkers.map((support) => (
          <div
            key={support.id}
            className="absolute bottom-6 flex translate-x-[-50%] flex-col items-center gap-1"
            style={{ left: `${2 + (support.percent * 0.96)}%` }}
            onContextMenu={(event) => openContextMenu(event, { kind: "support", id: support.id, x: support.position })}
          >
            <div
              className="cursor-ew-resize"
              onPointerDown={beginDrag({ type: "support", id: support.id, pointerId: 0 })}
              title={support.reaction ? `Düşey=${support.reaction.vertical.toFixed(2)} kN${Math.abs(support.reaction.axial) > 1e-3 ? `, Yatay=${support.reaction.axial.toFixed(2)} kN` : ""}` : ""}
            >
              {/* Larger support symbol */}
              <div className="h-0 w-0 border-x-[14px] border-b-[22px] border-x-transparent border-b-cyan-400" />
              <div className="mx-auto h-1.5 w-7 rounded-b bg-slate-600" />
            </div>
            <span className="text-xs font-semibold text-cyan-200">{support.id}</span>
          </div>
        ))}

        {/* Simplified point loads */}
        {pointHandles.map((load) => (
          <div
            key={load.id}
            className="absolute top-2 flex translate-x-[-50%] flex-col items-center gap-1"
            style={{ left: `${2 + (load.percent * 0.96)}%` }}
            onContextMenu={(event) => openContextMenu(event, { kind: "point", id: load.id, x: load.position })}
          >
            <div
              className="cursor-ew-resize"
              onPointerDown={beginDrag({ type: "point", id: load.id, pointerId: 0 })}
            >
              {/* Simplified arrow */}
              <svg
                className="h-8 w-8 text-red-500"
                viewBox="0 0 24 32"
                style={{ transform: `rotate(${load.angleDeg}deg)` }}
              >
                <line x1="12" y1="2" x2="12" y2="22" stroke="currentColor" strokeWidth="2" />
                <polygon points="6,22 18,22 12,30" fill="currentColor" />
              </svg>
            </div>
            <span className="rounded bg-red-500/20 px-1 py-0.5 text-xs text-red-200">
              {load.id}: {load.magnitude.toFixed(1)}kN
            </span>
          </div>
        ))}

        {/* UDL spans rebuilt: draggable start/end + whole-span center drag + center direction/magnitude */}
        {udlSpans.map((load) => {
          const startPos = 2 + (load.startPercent * 0.96);
          const width = (load.endPercent - load.startPercent) * 0.96;
          const endPos = startPos + width;
          return (
            <div
              key={load.id}
              className="absolute inset-x-0 top-8 h-14 z-30"
              onContextMenu={(event) => openContextMenu(event, { kind: "udl", id: load.id, x: (load.start + load.end) / 2 })}
            >
              {/* UDL bar - daha belirgin arka plan */}
              <div
                className="absolute h-6 rounded bg-orange-500 cursor-grab hover:bg-orange-400 border-2 border-orange-600 shadow-lg"
                style={{ left: `${startPos}%`, width: `${width}%` }}
                onPointerDown={beginDrag({ type: "udl-center", id: load.id, pointerId: 0 })}
              />

              {/* Start drag handle - daha büyük ve belirgin */}
              <div
                className="absolute -top-1 h-8 w-4 translate-x-[-50%] cursor-ew-resize rounded bg-orange-300 hover:bg-orange-200 border border-orange-500 shadow-md"
                style={{ left: `${startPos}%` }}
                onPointerDown={beginDrag({ type: "udl-start", id: load.id, pointerId: 0 })}
                title="Yayılı yük başlangıcını sürükle"
              />

              {/* End drag handle - daha büyük ve belirgin */}
              <div
                className="absolute -top-1 h-8 w-4 translate-x-[-50%] cursor-ew-resize rounded bg-orange-300 hover:bg-orange-200 border border-orange-500 shadow-md"
                style={{ left: `${endPos}%` }}
                onPointerDown={beginDrag({ type: "udl-end", id: load.id, pointerId: 0 })}
                title="Yayılı yük bitişini sürükle"
              />

              {/* Merkezde yön ve üstte kuvvet etiketi */}
              <div
                className="absolute pointer-events-none flex flex-col items-center"
                style={{ left: `${startPos + width / 2}%`, transform: "translateX(-50%)", top: "-6px" }}
              >
                <span className="mb-1 rounded bg-orange-600/80 px-2 py-0.5 text-sm font-semibold text-orange-100">
                  {load.magnitude.toFixed(1)} kN/m
                </span>
                <span className="text-lg font-bold text-orange-900 drop-shadow">
                  {load.direction === "down" ? "↓" : "↑"}
                </span>
              </div>
            </div>
          );
        })}

        {/* Simplified moment loads */}
        {momentHandles.map((moment) => (
          <div
            key={moment.id}
            className="absolute top-2 flex translate-x-[-50%] flex-col items-center gap-1"
            style={{ left: `${2 + (moment.percent * 0.96)}%` }}
            onContextMenu={(event) => openContextMenu(event, { kind: "moment", id: moment.id, x: moment.position })}
          >
            <div
              className="cursor-ew-resize"
              onPointerDown={beginDrag({ type: "moment", id: moment.id, pointerId: 0 })}
            >
              <div
                className={clsx(
                  "flex h-6 w-6 items-center justify-center rounded-full border text-xs font-bold",
                  moment.direction === "ccw" ? "border-emerald-300 text-emerald-200" : "border-rose-300 text-rose-200",
                )}
              >
                {moment.direction === "ccw" ? "↻" : "↺"}
              </div>
            </div>
            <span className="rounded bg-slate-800/70 px-1 py-0.5 text-xs text-slate-200">
              {moment.id}: {moment.magnitude.toFixed(1)}kN·m
            </span>
          </div>
        ))}

        {/* Basit uç etiketleri (biraz daha büyük) */}
        <div className="absolute bottom-2 left-[2%] right-[2%] flex justify-between text-sm text-slate-500">
          <span>0 m</span>
          <span>{beamLength.toFixed(1)} m</span>
        </div>

        {/* Dinamik parça mesafeleri (komşu noktalar arası) */}
        {dimensionSegments.map((seg, idx) => {
          const mid = (seg.start + seg.end) / 2;
          const midPercent = positionToPercent(mid);
          const text = `${(seg.end - seg.start).toFixed(2)} m`;
          return (
            <span
              key={`dim-${idx}`}
              className="absolute text-sm font-medium text-slate-300"
              style={{ left: `${2 + midPercent * 0.96}%`, transform: "translateX(-50%)", top: "calc(50% - 28px)" }}
            >
              {text}
            </span>
          );
        })}
      </div>
    </div>
  );
}
