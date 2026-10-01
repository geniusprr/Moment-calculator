"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { simulateBeam } from "@/lib/api";
import { interpolate, stepDeflection } from "@/lib/simulation";
import { HelpHint } from "@/components/HelpHint";
import type { BeamSimulationResponse, BeamSolveRequest, BeamSolveResponse } from "@/types/beam";

export function BeamSimulation({ payload, result, loading, error }: {
  payload: BeamSolveRequest; result: BeamSolveResponse | null; loading: boolean; error: string | null;
}) {
  const [open, setOpen] = useState(false);
  const [mode, setMode] = useState<"static" | "dynamic">("static");
  const [model, setModel] = useState<BeamSimulationResponse | null>(null);
  const [mass, setMass] = useState(100);
  const [damping, setDamping] = useState(2);
  const [busy, setBusy] = useState(false);
  const [simulationError, setSimulationError] = useState<string | null>(null);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(0.1);
  const [scale, setScale] = useState<number | null>(null);
  const [factor, setFactor] = useState(1);
  const [probe, setProbe] = useState<number | null>(null);
  const controller = useRef<AbortController | null>(null);
  const payloadKey = JSON.stringify(payload);

  useEffect(() => {
    controller.current?.abort();
    setModel(null); setBusy(false); setPlaying(false); setTime(0); setSimulationError(null);
    return () => controller.current?.abort();
  }, [payloadKey, mass, damping]);

  useEffect(() => {
    if (!open || !playing || !model || mode !== "dynamic") return;
    let frame: number;
    let last = performance.now();
    const tick = (now: number) => {
      const elapsed = Math.min((now-last)/1000, 0.1)*speed;
      last = now;
      setTime((t) => t+elapsed);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [open, playing, model, mode, speed]);

  const x = mode === "dynamic" && model ? model.x : result?.diagram.x ?? [];
  const staticY = useMemo(() => mode === "dynamic" && model ? model.static_deflection_mm : result?.diagram.deflection ?? [], [mode, model, result]);
  const y = useMemo(() => mode === "dynamic" && model ? stepDeflection(model, time) : staticY.map((v) => v*factor), [mode, model, time, staticY, factor]);
  const staticPeak = Math.max(0, ...staticY.map(Math.abs));
  // Bound all modal oscillations, so auto-scale never clips a transient.
  const envelope = model && mode === "dynamic" ? Math.max(0, ...model.x.map((_, j) =>
    model.modal_static_shapes_mm.reduce((sum, shape) => sum+2.1*Math.abs(shape[j]), 0))) : staticPeak;
  const autoScale = envelope > 1e-8 ? Math.min(10000, Math.max(0.001, 0.12*payload.length*1000/envelope)) : 1;
  const magnification = scale ?? autoScale;
  const sx = (v: number) => 64+v/payload.length*772;
  const sy = (v: number) => 190+v/1000*(772/payload.length)*magnification;
  const path = (values: number[]) => x.map((v, i) => `${i ? "L" : "M"}${sx(v).toFixed(3)},${sy(values[i] || 0).toFixed(3)}`).join(" ");
  const peakIndex = y.reduce((best, v, i) => Math.abs(v) > Math.abs(y[best] ?? 0) ? i : best, 0);
  const probeX = probe ?? x[peakIndex] ?? 0;
  const probeY = interpolate(x, y, probeX);
  const valid = !!result && !loading && !error;
  const hasData = valid && (mode === "static" || !!model);

  async function start() {
    controller.current?.abort();
    const next = new AbortController(); controller.current = next;
    setBusy(true); setPlaying(false); setSimulationError(null); setTime(0);
    try {
      const response = await simulateBeam({ ...payload, mass_per_length_kgm: mass, damping_ratio: damping/100 }, next.signal);
      if (!next.signal.aborted) { setModel(response); setPlaying(true); }
    } catch (err) {
      if (!next.signal.aborted) setSimulationError(err instanceof Error ? err.message : "Simülasyon hesaplanamadı.");
    } finally { if (!next.signal.aborted) setBusy(false); }
  }

  return <section className="panel simulation-panel overflow-hidden" aria-label="Kiriş sehim simülasyonu">
    <div className="flex flex-wrap items-center justify-between gap-3 p-5">
      <div><p className="eyebrow">DEFORMASYON LABORATUVARI</p><h2 className="mt-1 text-lg font-semibold">Kirişin davranışını gör</h2>
        <p className="mt-1 text-xs text-slate-400">Gerçek sehim değerleri · yukarı ve aşağı yer değiştirme</p></div>
      <button className="primary-button" type="button" aria-expanded={open} onClick={() => { setOpen(!open); setPlaying(false); }}>
        {open ? "Simülasyonu kapat" : "Kirişi simüle et"}<span aria-hidden>↗</span>
      </button>
    </div>
    {open && <div className="border-t border-slate-800 p-4 sm:p-5">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div className="segmented-control">
          <button type="button" aria-pressed={mode === "static"} onClick={() => { setMode("static"); setPlaying(false); }}>Statik sehim</button>
          <button type="button" aria-pressed={mode === "dynamic"} onClick={() => { setMode("dynamic"); setPlaying(false); }}>Dinamik titreşim</button>
        </div>
        <span className="text-xs text-slate-400">{mode === "static" ? "E · I · mesnet · yük" : "Mü̈ + Cũ + Ku = F"}
          <HelpHint label="Fizik modeli">Sabit kesitli doğrusal elastik Euler–Bernoulli kiriş modeli. Statik sehim Python motorundan gelir. Dinamik tepki tutarlı kütle matrisi ve tüm sonlu eleman modlarıyla hesaplanır. Çatlama, plastisite ve büyük deformasyonlar bu modelin kapsamı dışındadır.</HelpHint>
        </span>
      </div>
      {!valid && <div className="status-note" role="status">{error || "Güncel model hesaplanıyor. Simülasyon için geçerli sonuç bekleniyor."}</div>}
      {mode === "dynamic" && <div className="simulation-settings mb-4 grid gap-3 sm:grid-cols-3">
        <label>Kütle (kg/m) <HelpHint label="Kütle">Kirişin ve birlikte hareket eden yüklerin toplam birim uzunluk kütlesi. Titreşim frekansını belirler. Öz ağırlık kuvveti otomatik eklenmez; istersen yayılı yük olarak tanımla.</HelpHint>
          <input type="number" min="0.1" max="100000" value={mass} onChange={(e) => setMass(Number(e.target.value))} /></label>
        <label>Sönüm (%) <HelpHint label="Sönüm">Her titreşim modu için kritik sönüm oranı. %0 sönümsüz, %2 örnek bir değerdir; malzeme ve bağlantıya göre belirlenmelidir.</HelpHint>
          <input type="number" min="0" max="30" step="0.5" value={damping} onChange={(e) => setDamping(Number(e.target.value))} /></label>
        <button className="primary-button self-end" type="button" disabled={!valid || busy || mass <= 0 || mass > 100000 || damping < 0 || damping > 30}
          onClick={start}>{busy ? "Fizik modeli çözülüyor…" : "Ani yükü uygula"}</button>
      </div>}
      {simulationError && <p className="status-note text-rose-300" role="alert">{simulationError}</p>}
      <div className="simulation-canvas relative">
        <div className="absolute left-4 top-3 z-10 flex flex-wrap gap-3 text-[11px] text-slate-400">
          <span><i className="legend-line" />Şekil değiştirmemiş kiriş</span><span><i className="legend-line active" />{mode === "dynamic" ? "Anlık yer değiştirme" : "Elastik eğri"}</span>
        </div>
        <svg viewBox="0 0 900 380" className="w-full" role="img" aria-label="Kirişin sehim eğrisi; pozitif değerler aşağı yönlüdür"
          onPointerMove={(e) => { const rect = e.currentTarget.getBoundingClientRect(); setProbe(Math.max(0, Math.min(payload.length, ((e.clientX-rect.left)/rect.width*900-64)/772*payload.length))); }}
          onPointerLeave={() => setProbe(null)}>
          <defs><pattern id="beam-lab-grid" width="30" height="30" patternUnits="userSpaceOnUse"><path d="M30 0H0V30" fill="none" stroke="#252b34" strokeWidth="0.7" /></pattern></defs>
          <rect width="900" height="380" fill="url(#beam-lab-grid)" />
          <line x1="64" x2="836" y1="190" y2="190" stroke="#64748b" strokeWidth="4" strokeDasharray="9 7" />
          {valid && mode === "dynamic" && model && <path d={path(staticY)} fill="none" stroke="#67727e" strokeWidth="2" strokeDasharray="4 5" />}
          {valid && (mode === "static" || model) && <path d={path(y)} fill="none" stroke="#67e8f9" strokeWidth="6" strokeLinecap="round" />}
          {payload.supports.map((s) => <g key={s.id} transform={`translate(${sx(s.position)},190)`}>
            {s.type === "fixed" ? <><rect x={s.position > payload.length/2 ? 0 : -12} y="-28" width="12" height="56" fill="#a5b3c3" /><path d="M-16 -25L-24 -17M-16 -10L-24 -2M-16 5L-24 13M-16 20L-24 28" stroke="#64748b" /></>
              : <><path d="M0 7L-13 27H13Z" fill="#131922" stroke="#b3bfcc" strokeWidth="2" />{s.type === "roller" && <><circle cx="-7" cy="32" r="3" fill="#b3bfcc"/><circle cx="7" cy="32" r="3" fill="#b3bfcc"/></>}</>}
            <text y="53" textAnchor="middle" fill="#aab5c2" fontSize="13">{s.id}</text>
          </g>)}
          {[0, 0.25, 0.5, 0.75, 1].map((part) => <text key={part} x={sx(part*payload.length)} y="353" textAnchor="middle" fill="#7f8b9c" fontSize="12">{(part*payload.length).toFixed(2)} m</text>)}
          {valid && (mode === "static" || model) && <><line x1={sx(probeX)} x2={sx(probeX)} y1="80" y2="325" stroke="#67e8f9" strokeOpacity="0.2" strokeDasharray="4 5" />
            <circle cx={sx(probeX)} cy={sy(probeY)} r="5" fill="#111820" stroke="#67e8f9" strokeWidth="2" /></>}
          <text x="867" y="132" fill="#7f8b9c" fontSize="16">↑ −</text><text x="867" y="265" fill="#7f8b9c" fontSize="16">↓ +</text>
        </svg>
        {mode === "dynamic" && !model && <div className="absolute inset-0 flex items-center justify-center pointer-events-none"><p className="rounded-xl bg-slate-900/90 px-4 py-3 text-sm text-slate-300">Kütle ve sönümü gir, ani yükü uygula.</p></div>}
      </div>
      <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <div className="simulation-stat"><span>{mode === "dynamic" ? "Anlık en büyük |w|" : "En büyük |w|"}</span><strong>{hasData ? Math.abs(y[peakIndex] || 0).toFixed(4) : "—"} <small>mm</small></strong></div>
        <div className="simulation-stat"><span>İncelenen nokta</span><strong>{hasData ? probeY.toFixed(4) : "—"} <small>mm</small></strong><span>x = {probeX.toFixed(3)} m</span></div>
        <div className="simulation-stat"><span>Görsel büyütme</span><strong>{magnification.toFixed(1)}<small>×</small></strong><span>Sayısal sehim büyütülmez</span></div>
        <div className="simulation-stat"><span>{mode === "dynamic" ? "Temel frekans" : "Yük katsayısı"}</span><strong>{mode === "dynamic" ? model?.fundamental_frequency_hz.toFixed(3) ?? "—" : (factor*100).toFixed(0)}<small>{mode === "dynamic" ? " Hz" : " %"}</small></strong></div>
      </div>
      <div className="mt-4 grid gap-4 sm:grid-cols-2 simulation-settings">
        <label>Görsel ölçek <HelpHint label="Görsel ölçek">Gerçek sehim genellikle çok küçüktür. Bu katsayı sadece çizimi büyütür. “Gerçek ölçek” ile uzunluk ve sehim aynı ölçekte çizilir.</HelpHint>
          <select value={scale === null ? "auto" : String(scale)} onChange={(e) => setScale(e.target.value === "auto" ? null : Number(e.target.value))}>
            <option value="auto">Otomatik · {autoScale.toFixed(1)}×</option><option value="1">Gerçek ölçek · 1×</option><option value="10">10×</option><option value="100">100×</option><option value="1000">1000×</option>
          </select></label>
        {mode === "static" ? <label>Yük katsayısı · {(factor*100).toFixed(0)}% <HelpHint label="Yük katsayısı">Tüm kuvvet ve momentleri birlikte ölçekler. Eksi değer yük yönlerini ters çevirir. Doğrusal elastik modelde sehim aynı katsayıyla değişir.</HelpHint>
          <input type="range" min="-1" max="1" step="0.01" value={factor} onChange={(e) => setFactor(Number(e.target.value))} /></label>
          : <label>Oynatma hızı · t = {time.toFixed(3)} s<select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
            <option value="0.02">0.02× · çok yavaş</option><option value="0.1">0.1× · yavaş</option><option value="1">1× · gerçek zaman</option>
          </select></label>}
      </div>
      {mode === "dynamic" && model && <div className="mt-4 flex flex-wrap gap-2">
        <button className="secondary-button" type="button" disabled={!valid} onClick={() => setPlaying(!playing)}>{playing ? "Duraklat" : "Devam et"}</button>
        <button className="secondary-button" type="button" onClick={() => { setTime(0); setPlaying(true); }}>Baştan oynat</button>
        <button className="secondary-button" type="button" onClick={() => { setPlaying(false); setTime(0); }}>t = 0</button>
        <span className="self-center text-xs text-slate-500">{model.element_count} eleman · {model.angular_frequencies_rad_s.length} mod · ani sabit yük</span>
      </div>}
      <p className="mt-4 text-[11px] leading-relaxed text-slate-500">Küçük deformasyon, sabit E ve I, doğrusal elastik davranış. Kütleye bağlı öz ağırlık otomatik eklenmez. Gösterilen değerler mm, konumlar m cinsindedir.</p>
    </div>}
  </section>;
}
