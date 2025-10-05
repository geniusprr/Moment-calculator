"use client";

import { CSSProperties, useCallback, useEffect, useMemo, useRef, useState } from "react";
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

  const UDL_AREA_HEIGHT = 64;
  const UDL_TOP = `calc(50% - ${15 + UDL_AREA_HEIGHT}px)`;
  const ORANGE_COLOR = "#f97316";
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
              style={{ transform: `rotate(${-(load.angleDeg + 90)}deg)` }}
            >
              <title>{`${load.id} açısı: ${load.angleDeg}°`}</title>
              <line x1="12" y1="0" x2="12" y2="54" stroke="currentColor" strokeWidth="2.5" />
              <polygon points="6,54 18,54 12,64" fill="currentColor" />
            </svg>
          </div>
        ))}

        {/* UDL spans with improved arrow alignment */}
        {udlSpans.map((load) => {
          const startPercent = 2 + load.startPercent * 0.96;
          const widthPercent = (load.endPercent - load.startPercent) * 0.96;
          const arrowCount = Math.max(3, Math.min(11, Math.round(widthPercent / 8)));
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
                className="absolute pointer-events-none"
                style={{
                  top: -32,
                  left: "50%",
                  transform: "translateX(-50%)",
                  whiteSpace: "nowrap",
                  maxWidth: "100%",
                }}
              >
                <span className="rounded bg-orange-600/90 px-2 py-0.5 text-xs font-semibold text-white">
                  {labelText}
                </span>
              </div>

              {load.shape === "uniform" ? (
                <div
                  className="absolute left-0 right-0 h-0.5"
                  style={{ ...baseLinePosition, backgroundColor: ORANGE_COLOR }}
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
                    stroke={ORANGE_COLOR}
                    strokeWidth="1.5"
                    vectorEffect="non-scaling-stroke"
                  />
                </svg>
              )}

              <div
                className="absolute inset-0 cursor-grab transition hover:opacity-90"
                onPointerDown={beginDrag({ type: "udl-center", id: load.id, pointerId: 0 })}
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
                      className="absolute text-orange-500"
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
                        stroke="currentColor"
                        strokeWidth="2"
                      />
                      <polygon
                        points={`${UDL_ARROW_HALF_WIDTH - scaledHeadWidth},${UDL_AREA_HEIGHT - scaledHeadHeight} ${UDL_ARROW_HALF_WIDTH + scaledHeadWidth},${UDL_AREA_HEIGHT - scaledHeadHeight} ${UDL_ARROW_HALF_WIDTH},${UDL_AREA_HEIGHT}`}
                        fill="currentColor"
                      />
                    </svg>
                  );
                })}
              </div>

              <div
                className="absolute z-40 w-8 translate-x-[-50%] cursor-ew-resize"
                style={{ left: "0%", top: "8px", height: `${UDL_AREA_HEIGHT - 16}px` }}
                onPointerDown={beginDrag({ type: "udl-start", id: load.id, pointerId: 0 })}
                title="Yayılı yük başlangıcını sürükle"
              />

              <div
                className="absolute z-40 w-8 translate-x-[-50%] cursor-ew-resize"
                style={{ left: "100%", top: "8px", height: `${UDL_AREA_HEIGHT - 16}px` }}
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
