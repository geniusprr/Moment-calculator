"use client";

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { simulateBeam } from "@/lib/api";
import { stepDeflection } from "@/lib/simulation";
import { HelpHint } from "@/components/HelpHint";
import type { BeamDeformation } from "@/components/BeamSketch";
import type { BeamSimulationResponse, BeamSolveRequest, BeamSolveResponse } from "@/types/beam";

export function BeamSimulation({ payload, result, loading, error, renderSketch, children }: {
  payload: BeamSolveRequest; result: BeamSolveResponse | null; loading: boolean; error: string | null;
  renderSketch: (deformation?: BeamDeformation) => ReactNode; children?: ReactNode;
}) {
  const [mode, setMode] = useState<"loads" | "static" | "dynamic">("loads");
  const [settings, setSettings] = useState(false);
  const [model, setModel] = useState<BeamSimulationResponse | null>(null);
  const [mass, setMass] = useState(100);
  const [damping, setDamping] = useState(2);
  const [busy, setBusy] = useState(false);
  const [simulationError, setSimulationError] = useState<string | null>(null);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(0.1);
  const [scale, setScale] = useState<number | null>(null);
  const controller = useRef<AbortController | null>(null);
  const payloadKey = JSON.stringify(payload);

  useEffect(() => {
    controller.current?.abort();
    setModel(null); setBusy(false); setPlaying(false); setTime(0); setSimulationError(null);
    return () => controller.current?.abort();
  }, [payloadKey, mass, damping]);

  useEffect(() => {
    if (!playing || !model || mode !== "dynamic") return;
    let frame: number;
    let last = performance.now();
    const tick = (now: number) => {
      const elapsed = Math.min((now - last) / 1000, 0.1) * speed;
      last = now;
      setTime((t) => t + elapsed);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing, model, mode, speed]);

  const valid = !!result && !loading && !error;
  const dynamic = mode === "dynamic";
  const x = dynamic && model ? model.x : result?.diagram.x ?? [];
  const staticY = useMemo(() => dynamic && model ? model.static_deflection_mm : result?.diagram.deflection ?? [], [dynamic, model, result]);
  const y = useMemo(() => dynamic && model ? stepDeflection(model, time) : staticY, [dynamic, model, time, staticY]);
  const envelope = useMemo(() => dynamic && model ? Math.max(0, ...model.x.map((_, j) =>
    model.modal_static_shapes_mm.reduce((sum, shape) => sum + 2.1 * Math.abs(shape[j]), 0))) : Math.max(0, ...staticY.map(Math.abs)), [dynamic, model, staticY]);
  const autoScale = envelope > 1e-8 ? Math.min(10000, Math.max(0.001, 0.10 * payload.length * 1000 / envelope)) : 1;
  const magnification = scale ?? autoScale;
  const hasData = valid && (!dynamic || !!model);
  const peak = Math.max(0, ...y.map(Math.abs));

  async function start() {
    controller.current?.abort();
    const next = new AbortController(); controller.current = next;
    setBusy(true); setPlaying(false); setSimulationError(null); setTime(0);
    try {
      const response = await simulateBeam({ ...payload, mass_per_length_kgm: mass, damping_ratio: damping / 100 }, next.signal);
      if (!next.signal.aborted) { setModel(response); setPlaying(true); setSettings(false); }
    } catch (err) {
      if (!next.signal.aborted) setSimulationError(err instanceof Error ? err.message : "Simülasyon hesaplanamadı.");
    } finally { if (!next.signal.aborted) setBusy(false); }
  }

  return <section className="beam-stage panel" aria-label="Kiriş çalışma alanı">
    <div className="beam-toolbar">
      <div className="segmented-control">
        {([['loads', 'Yükler'], ['static', 'Sehim'], ['dynamic', 'Simüle et']] as const).map(([value, label]) => <button key={value} type="button" aria-pressed={mode === value} onClick={() => { setMode(value); setPlaying(false); }}>{label}</button>)}
      </div>
      <button type="button" className="secondary-button" aria-expanded={settings} onClick={() => setSettings(!settings)}>Ayarlar</button>
    </div>
    {renderSketch(hasData && mode !== "loads" ? { x, valuesMm: y, magnification } : undefined)}
    <div className="beam-readout" role="status">
      <span>{error ? "Modeli kontrol et" : loading ? "Hesaplanıyor…" : mode === "loads" ? "Yükleri sürükleyerek düzenle" : hasData ? `|w|max = ${peak.toFixed(4)} mm · çizim ${magnification.toFixed(1)}×` : "Ani yükü uygula"}</span>
      {dynamic && model && valid && <span>{model.fundamental_frequency_hz.toFixed(2)} Hz · {time.toFixed(2)} s</span>}
      <HelpHint label="Kiriş modeli">Pozitif sehim aşağı, negatif sehim yukarı yönlüdür. Görsel büyütme yalnızca çizimi etkiler; mm değerleri gerçektir. Sabit E ve I ile doğrusal elastik Euler–Bernoulli modeli; dinamik çözüm tutarlı kütle matrisi ve tüm sonlu eleman modlarını kullanır. Büyük deformasyon, çatlama ve plastisite kapsam dışındadır.</HelpHint>
    </div>
    {dynamic && <div className="beam-playback">
      <button className="primary-button" type="button" disabled={!valid || busy || !Number.isFinite(mass) || mass <= 0 || mass > 100000 || !Number.isFinite(damping) || damping < 0 || damping > 30} onClick={start}>{busy ? "Çözülüyor…" : model ? "Baştan uygula" : "Ani yükü uygula"}</button>
      {model && <button className="secondary-button" type="button" disabled={!valid} onClick={() => setPlaying(!playing)}>{playing ? "Duraklat" : "Devam et"}</button>}
    </div>}
    {settings && <div className="beam-settings simulation-settings">
      <div className="grid grid-cols-2 gap-3">
        <label>Görsel ölçek<select value={scale === null ? "auto" : String(scale)} onChange={(e) => setScale(e.target.value === "auto" ? null : Number(e.target.value))}><option value="auto">Otomatik · {autoScale.toFixed(1)}×</option><option value="1">Gerçek ölçek · 1×</option><option value="10">10×</option><option value="100">100×</option><option value="1000">1000×</option></select></label>
        <label>Oynatma hızı<select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}><option value="0.02">0.02×</option><option value="0.1">0.1×</option><option value="1">Gerçek zaman</option></select></label>
        <label>Kütle (kg/m)<HelpHint label="Kütle">Birlikte hareket eden toplam birim uzunluk kütlesi. Öz ağırlık otomatik eklenmez; gerekirse yayılı yük tanımla.</HelpHint><input type="number" min="0.1" max="100000" value={mass} onChange={(e) => setMass(Number(e.target.value))} /></label>
        <label>Sönüm (%)<HelpHint label="Sönüm">Her mod için kritik sönüm oranı. %2 örnek değerdir; malzeme ve bağlantıya göre belirle.</HelpHint><input type="number" min="0" max="30" step="0.5" value={damping} onChange={(e) => setDamping(Number(e.target.value))} /></label>
      </div>
    </div>}
    {simulationError && <p className="status-note" role="alert">{simulationError}</p>}
    {children}
  </section>;
}
