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
      <div ref={containerRef} className="relative h-56 select-none rounded-xl border border-slate-800/60 bg-slate-900/60">
        {/* Simplified beam bar - positioned in middle */}
        <div className="absolute left-[2%] right-[2%] h-3 rounded bg-slate-200/80 shadow-sm" style={{ top: 'calc(50% - 15px)' }} />

        {/* Simplified support markers - below the beam, tip touching it */}
        {supportMarkers.map((support) => (
          <div
            key={support.id}
            className="absolute flex translate-x-[-50%] flex-col items-center gap-1"
            style={{ left: `${2 + (support.percent * 0.96)}%`, top: 'calc(50% - 3px)' }}
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

        {/* Simplified point loads - above the beam, touching it */}
        {pointHandles.map((load) => (
          <div
            key={load.id}
            className="absolute flex translate-x-[-50%] flex-col items-center cursor-ew-resize z-40"
            style={{ left: `${2 + (load.percent * 0.96)}%`, bottom: 'calc(50% + 18px)' }}
            onContextMenu={(event) => openContextMenu(event, { kind: "point", id: load.id, x: load.position })}
            onPointerDown={beginDrag({ type: "point", id: load.id, pointerId: 0 })}
          >
            <span className="mb-0.5 rounded bg-red-500/90 px-1.5 py-0.5 text-xs font-semibold text-white">
              {load.id}: {load.magnitude.toFixed(1)}kN
            </span>
            {/* Arrow pointing down towards the beam */}
            <svg
              className="h-16 w-6 text-red-500"
              viewBox="0 0 24 64"
              style={{ transform: `rotate(${load.angleDeg + 90}deg)` }}
            >
              <line x1="12" y1="0" x2="12" y2="54" stroke="currentColor" strokeWidth="2.5" />
              <polygon points="6,54 18,54 12,64" fill="currentColor" />
            </svg>
          </div>
        ))}

        {/* UDL spans rebuilt: draggable start/end + whole-span center drag + arrows touching beam */}
        {udlSpans.map((load) => {
          const startPos = 2 + (load.startPercent * 0.96);
          const width = (load.endPercent - load.startPercent) * 0.96;
          const endPos = startPos + width;
          const arrowCount = Math.max(3, Math.min(8, Math.floor(width * 0.5))); // Daha az ok
          return (
            <div
              key={load.id}
              className="absolute inset-x-0 z-30"
              style={{ bottom: 'calc(50% + 15px)', height: '70px' }}
              onContextMenu={(event) => openContextMenu(event, { kind: "udl", id: load.id, x: (load.start + load.end) / 2 })}
            >
              {/* Magnitude label at top */}
              <div
                className="absolute pointer-events-none"
                style={{ left: `${startPos + width / 2}%`, transform: "translateX(-50%)", top: "0px" }}
              >
                <span className="rounded bg-orange-600/90 px-2 py-0.5 text-xs font-semibold text-white">
                  {load.magnitude.toFixed(1)} kN/m
                </span>
              </div>

              {/* Horizontal line at top connecting arrows */}
              <div
                className="absolute h-0.5 bg-orange-500"
                style={{ left: `${startPos}%`, width: `${width}%`, top: '28px' }}
              />

              {/* Multiple arrows representing distributed load */}
              <div
                className="absolute cursor-grab hover:opacity-80"
                style={{ left: `${startPos}%`, width: `${width}%`, top: '28px', height: '42px' }}
                onPointerDown={beginDrag({ type: "udl-center", id: load.id, pointerId: 0 })}
              >
                {Array.from({ length: arrowCount }).map((_, i) => {
                  const position = (i / (arrowCount - 1)) * 100;
                  return (
                    <svg
                      key={i}
                      className="absolute text-orange-500"
                      style={{ left: `${position}%`, transform: 'translateX(-50%)', top: '0' }}
                      width="16"
                      height="42"
                      viewBox="0 0 16 42"
                    >
                      <line x1="8" y1="0" x2="8" y2="34" stroke="currentColor" strokeWidth="2" />
                      <polygon points="4,34 12,34 8,42" fill="currentColor" />
                    </svg>
                  );
                })}
              </div>

              {/* Invisible start drag handle */}
              <div
                className="absolute h-10 w-8 translate-x-[-50%] cursor-ew-resize z-10"
                style={{ left: `${startPos}%`, top: '28px' }}
                onPointerDown={beginDrag({ type: "udl-start", id: load.id, pointerId: 0 })}
                title="Yayılı yük başlangıcını sürükle"
              />

              {/* Invisible end drag handle */}
              <div
                className="absolute h-10 w-8 translate-x-[-50%] cursor-ew-resize z-10"
                style={{ left: `${endPos}%`, top: '28px' }}
                onPointerDown={beginDrag({ type: "udl-end", id: load.id, pointerId: 0 })}
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
            style={{ left: `${2 + (moment.percent * 0.96)}%`, bottom: 'calc(50% + 21px)' }}
            onContextMenu={(event) => openContextMenu(event, { kind: "moment", id: moment.id, x: moment.position })}
            onPointerDown={beginDrag({ type: "moment", id: moment.id, pointerId: 0 })}
          >
            <span className="rounded bg-slate-800/90 px-2 py-0.5 text-xs font-semibold text-slate-200 pointer-events-none">
              {moment.magnitude.toFixed(1)} kN·m
            </span>
            <div
              className={clsx(
                "flex h-8 w-8 items-center justify-center rounded-full border-2 text-lg font-bold",
                moment.direction === "ccw" ? "border-emerald-300 text-emerald-200" : "border-rose-300 text-rose-200",
              )}
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
                style={{ left: `${2 + startPercent * 0.96}%`, top: "calc(50% + 45px)" }}
              />
              {/* End vertical line */}
              <div
                className="absolute w-0.5 h-8 bg-slate-400"
                style={{ left: `${2 + endPercent * 0.96}%`, top: "calc(50% + 45px)" }}
              />
              {/* Horizontal dimension line */}
              <div
                className="absolute h-0.5 bg-slate-400"
                style={{ 
                  left: `${2 + startPercent * 0.96}%`, 
                  width: `${(endPercent - startPercent) * 0.96}%`,
                  top: "calc(50% + 53px)"
                }}
              />
              {/* Distance text */}
              <span
                className="absolute text-xs font-medium text-slate-300 bg-slate-800/80 px-1.5 py-0.5 rounded"
                style={{ left: `${2 + midPercent * 0.96}%`, transform: "translateX(-50%)", top: "calc(50% + 58px)" }}
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
