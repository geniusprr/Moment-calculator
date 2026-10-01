"use client";

import { useState } from "react";
import { HelpHint } from "@/components/HelpHint";

import type {
  BeamType,
  MomentLoadInput,
  PointLoadInput,
  SupportInput,
  UdlInput,
} from "@/types/beam";

interface BeamFormProps {
  beamType: BeamType;
  length: number;
  onLengthChange: (value: number) => void;
  supports: SupportInput[];
  onSupportChange: (id: string, field: keyof SupportInput, value: string | number) => void;
  onAddSupport: () => void;
  onRemoveSupport: (id: string) => void;
  pointLoads: PointLoadInput[];
  onPointLoadChange: (id: string, field: keyof PointLoadInput, value: string | number) => void;
  onAddPointLoad: () => void;
  onRemovePointLoad: (id: string) => void;
  udls: UdlInput[];
  onUdlChange: (id: string, field: keyof UdlInput, value: string | number) => void;
  onAddUdl: () => void;
  onRemoveUdl: (id: string) => void;
  momentLoads: MomentLoadInput[];
  onMomentChange: (id: string, field: keyof MomentLoadInput, value: string | number) => void;
  onAddMoment: () => void;
  onRemoveMoment: (id: string) => void;
  elasticModulusGpa: number;
  onElasticModulusGpaChange: (value: number) => void;
  momentInertiaCm4: number;
  onMomentInertiaCm4Change: (value: number) => void;
  onReset: () => void;
  disableSolveReason: string | null;
}

const fieldClasses =
  "w-full rounded-xl border border-slate-700/80 bg-slate-900/80 px-4 py-2 text-sm text-slate-100 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-500/40";

const sectionTitleClass = "text-sm font-semibold text-slate-200";
const sectionHintClass = "text-xs leading-relaxed text-slate-400";
const labelClass = "text-xs font-medium text-slate-400";

export function BeamForm({
  beamType,
  length,
  onLengthChange,
  supports,
  onSupportChange,
  onAddSupport,
  onRemoveSupport,
  pointLoads,
  onPointLoadChange,
  onAddPointLoad,
  onRemovePointLoad,
  udls,
  onUdlChange,
  onAddUdl,
  onRemoveUdl,
  momentLoads,
  onMomentChange,
  onAddMoment,
  onRemoveMoment,
  onReset,
  disableSolveReason,
  elasticModulusGpa,
  onElasticModulusGpaChange,
  momentInertiaCm4,
  onMomentInertiaCm4Change,
}: BeamFormProps) {
  const maxSupports = beamType === "cantilever" ? 4 : 2;
  const [width, setWidth] = useState(30);
  const [height, setHeight] = useState(50);
  const supportHint =
    beamType === "cantilever"
      ? "Konsol çözümü için en az bir ankastre mesnet gerekir (x=0 veya x=L). Ek mesnet ekleyebilirsiniz."
      : "Basit kiriş için iki farklı konumda mafsallı veya kayar mesnet tanımlayın.";
  return (
    <div className="panel beam-form space-y-6 p-5">
      <div className="flex items-center justify-between">
        <div>
          <p className="eyebrow">01 / MODEL GİRİŞLERİ</p>
          <h3 className="mt-1 text-lg font-semibold">Kirişini tanımla</h3>
          <p className="mt-1 text-xs text-slate-400">Birimler her alanın yanında gösterilir.</p>
        </div>
        <button
          onClick={onReset}
          className="rounded-full border border-slate-700/70 px-3 py-1 text-xs text-slate-300 transition hover:border-slate-500 hover:text-white"
          type="button"
        >
          Sıfırla
        </button>
      </div>

      <section className="grid grid-cols-2 gap-4">
        <label className="col-span-2 space-y-2">
          <span className={labelClass}>Kiriş uzunluğu L (m) <HelpHint label="Kiriş uzunluğu">Toplam kiriş boyu. Tüm konumlar sol uçtan ölçülür: x = 0 sol uç, x = L sağ uç. 0.5 m’den büyük ve en fazla 30 m girin.</HelpHint></span>
          <input
            type="number"
            min={0.6}
            max={30}
            step={0.1}
            className={fieldClasses}
            value={length}
            onChange={(event) => onLengthChange(Number(event.target.value))}
          />
        </label>
        <label className="space-y-2">
          <span className={labelClass}>Elastisite E (GPa) <HelpHint label="Elastisite modülü">Malzemenin elastik rijitliği. Çelik için 200 GPa örnek bir değerdir. E arttıkça sehim azalır. Beton için proje/malzeme değerini kullanın.</HelpHint></span>
          <input
            type="number"
            min={1}
            max={1000}
            step={1}
            className={fieldClasses}
            value={elasticModulusGpa}
            onChange={(event) => onElasticModulusGpaChange(Number(event.target.value))}
          />
        </label>
        <label className="space-y-2">
          <span className={labelClass}>Atalet I (cm⁴) <HelpHint label="Atalet momenti">Eğilme eksenine göre kesit atalet momenti. Dikdörtgen kesitte I = b · h³ / 12. h, düşey kesit yüksekliğidir. 1 cm⁴ = 10⁻⁸ m⁴.</HelpHint></span>
          <input
            type="number"
            min={1}
            max={1000000}
            step={10}
            className={fieldClasses}
            value={momentInertiaCm4}
            onChange={(event) => onMomentInertiaCm4Change(Number(event.target.value))}
          />
        </label>
      </section>
      <details className="section-helper">
        <summary>Dikdörtgen kesitten I hesapla</summary>
        <div className="mt-3 grid grid-cols-2 gap-3">
          <label className="text-xs text-slate-400">Genişlik b (cm)<input className={`${fieldClasses} mt-1`} type="number" min="0.1" value={width} onChange={(e) => setWidth(Number(e.target.value))} /></label>
          <label className="text-xs text-slate-400">Yükseklik h (cm)<input className={`${fieldClasses} mt-1`} type="number" min="0.1" value={height} onChange={(e) => setHeight(Number(e.target.value))} /></label>
          <p className="col-span-2 text-xs text-slate-400">I = b × h³ / 12 = {(width*height**3/12).toLocaleString("tr-TR", { maximumFractionDigits: 2 })} cm⁴</p>
          <button className="secondary-button col-span-2" type="button" disabled={width <= 0 || height <= 0} onClick={() => onMomentInertiaCm4Change(width*height**3/12)}>Bu I değerini kullan</button>
        </div>
      </details>

      <section className="space-y-3">
        <header className="flex items-center justify-between">
          <div>
            <p className={sectionTitleClass}>02 / Mesnetler <HelpHint label="Mesnetler">Mafsallı ve kayar mesnetlerde düşey yer değiştirme sıfırdır; dönme serbesttir. Ankastre mesnette hem sehim hem dönme sıfırdır. Konsolda en az bir uç ankastre olmalıdır.</HelpHint></p>
            <p className={sectionHintClass}>{supportHint}</p>
          </div>
          <button
            type="button"
            onClick={onAddSupport}
            disabled={supports.length >= maxSupports}
            className="rounded-full bg-slate-800/70 px-3 py-1 text-xs font-semibold text-slate-200 transition hover:bg-slate-700/80 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Mesnet ekle
          </button>
        </header>
        {supports.length === 0 ? (
          <div className="panel-muted p-4 text-sm text-slate-400">
            {beamType === "cantilever"
              ? "Konsol için ankastre mesnet ekleyin."
              : "Çözümü etkinleştirmek için iki mesnet ekleyin."}
          </div>
        ) : (
          <div className="space-y-3">
            {supports.map((support) => (
              <div key={support.id} className="panel-muted relative flex flex-col gap-3 border border-slate-800/60 p-4">
                <button
                  type="button"
                  onClick={() => onRemoveSupport(support.id)}
                  className="absolute right-2 top-2 flex h-6 w-6 items-center justify-center rounded-full bg-rose-500/20 text-rose-300 transition hover:bg-rose-500/40 hover:text-rose-200"
                  title="Kaldır"
                >
                  ×
                </button>
                <div className="grid gap-3 sm:grid-cols-3">
                  <label className="space-y-1">
                    <span className={labelClass}>Etiket</span>
                    <input
                      type="text"
                      className={fieldClasses}
                      value={support.id}
                      onChange={(event) => onSupportChange(support.id, "id", event.target.value.toUpperCase())}
                    />
                  </label>
                  <label className="space-y-1">
                    <span className={labelClass}>Tip</span>
                    <select
                      className={fieldClasses}
                      value={support.type}
                      onChange={(event) => onSupportChange(support.id, "type", event.target.value)}
                      disabled={beamType === "cantilever"}
                    >
                      <option value="pin">Menteşe</option>
                      <option value="roller">Kayar</option>
                      {beamType === "cantilever" && <option value="fixed">Ankastre</option>}
                    </select>
                  </label>
                  <label className="space-y-1">
                    <span className={labelClass}>Konum (m)</span>
                    <input
                      type="number"
                      min={0}
                      max={length}
                      step={0.1}
                      className={fieldClasses}
                      value={support.position}
                      onChange={(event) => onSupportChange(support.id, "position", Number(event.target.value))}
                    />
                  </label>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="space-y-3">
        <header className="flex items-center justify-between">
          <p className={sectionTitleClass}>03 / Tekil yükler <HelpHint label="Tekil yük">Kuvveti kN, konumu m olarak girin. Açı −90° aşağı, +90° yukarı, 0° sağ yönüdür. Yük yönü sehim yönünü belirler.</HelpHint></p>
          <button
            type="button"
            onClick={onAddPointLoad}
            className="rounded-full bg-cyan-500/20 px-3 py-1 text-xs font-semibold text-cyan-200 transition hover:bg-cyan-500/40"
          >
            Tekil yük ekle
          </button>
        </header>
        {pointLoads.length === 0 ? (
          <div className="panel-muted p-4 text-sm text-slate-400">Büyüklük, konum ve açı ile konsantre kuvvetler ekleyin.</div>
        ) : (
          <div className="space-y-3">
            {pointLoads.map((load) => (
              <div key={load.id} className="panel-muted relative flex flex-col gap-3 border border-slate-800/60 p-4">
                <button
                  type="button"
                  onClick={() => onRemovePointLoad(load.id)}
                  className="absolute right-2 top-2 flex h-6 w-6 items-center justify-center rounded-full bg-rose-500/20 text-rose-300 transition hover:bg-rose-500/40 hover:text-rose-200"
                  title="Kaldır"
                >
                  ×
                </button>
                <div className="grid gap-3 sm:grid-cols-2">
                  <label className="space-y-1">
                    <span className={labelClass}>Etiket</span>
                    <input
                      type="text"
                      className={fieldClasses}
                      value={load.id}
                      onChange={(event) => onPointLoadChange(load.id, "id", event.target.value.toUpperCase())}
                    />
                  </label>
                  <label className="space-y-1">
                    <span className={labelClass}>Büyüklük (kN)</span>
                    <input
                      type="number"
                      min={0}
                      step={0.1}
                      className={fieldClasses}
                      value={load.magnitude}
                      onChange={(event) => onPointLoadChange(load.id, "magnitude", Number(event.target.value))}
                    />
                  </label>
                </div>
                <div className="grid gap-3 sm:grid-cols-2">
                  <label className="space-y-1">
                    <span className={labelClass}>Konum (m)</span>
                    <input
                      type="number"
                      min={0}
                      max={length}
                      step={0.1}
                      className={fieldClasses}
                      value={load.position}
                      onChange={(event) => onPointLoadChange(load.id, "position", Number(event.target.value))}
                    />
                  </label>
                  <label className="space-y-1">
                    <span className={labelClass}>Açı (derece)</span>
                    <input
                      type="number"
                      min={-180}
                      max={180}
                      step={5}
                      className={fieldClasses}
                      value={load.angleDeg}
                      onChange={(event) => onPointLoadChange(load.id, "angleDeg", Number(event.target.value))}
                    />
                    <p className="text-[11px] text-slate-500">0 = sağ, -90 = aşağı, 90 = yukarı</p>
                  </label>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="space-y-3">
        <header className="flex items-center justify-between">
          <p className={sectionTitleClass}>04 / Yayılı yükler <HelpHint label="Yayılı yük">Yoğunluk kN/m cinsindedir. Başlangıç ve bitiş sol uçtan ölçülür. Üçgen yüklerde büyüklük tepe yoğunluğudur; toplam kuvvet q × aralık / 2 olur.</HelpHint></p>
          <button
            type="button"
            onClick={onAddUdl}
            className="rounded-full bg-indigo-500/20 px-3 py-1 text-xs font-semibold text-indigo-200 transition hover:bg-indigo-500/40"
          >
            Yayılı yük ekle
          </button>
        </header>
        {udls.length === 0 ? (
          <div className="panel-muted p-4 text-sm text-slate-400">Yoğunluk ve açıklık ile dağıtılmış yükler tanımlayın.</div>
        ) : (
          <div className="space-y-3">
            {udls.map((load) => (
              <div key={load.id} className="panel-muted relative flex flex-col gap-3 border border-slate-800/60 p-4">
                <button
                  type="button"
                  onClick={() => onRemoveUdl(load.id)}
                  className="absolute right-2 top-2 flex h-6 w-6 items-center justify-center rounded-full bg-rose-500/20 text-rose-300 transition hover:bg-rose-500/40 hover:text-rose-200"
                  title="Kaldır"
                >
                  ×
                </button>
                <div className="grid gap-3 sm:grid-cols-2">
                  <label className="space-y-1">
                    <span className={labelClass}>Etiket</span>
                    <input
                      type="text"
                      className={fieldClasses}
                      value={load.id}
                      onChange={(event) => onUdlChange(load.id, "id", event.target.value.toUpperCase())}
                    />
                  </label>
                  <label className="space-y-1">
                    <span className={labelClass}>Yoğunluk (kN/m)</span>
                    <input
                      type="number"
                      min={0}
                      step={0.1}
                      className={fieldClasses}
                      value={load.magnitude}
                      onChange={(event) => onUdlChange(load.id, "magnitude", Number(event.target.value))}
                    />
                  </label>
                </div>
                <div className="grid gap-3">
                  <label className="space-y-1">
                    <span className={labelClass}>Başlangıç&nbsp;(m)</span>
                    <input
                      type="number"
                      min={0}
                      max={length}
                      step={0.1}
                      className={fieldClasses}
                      value={load.start}
                      onChange={(event) => onUdlChange(load.id, "start", Number(event.target.value))}
                    />
                  </label>
                  <label className="space-y-1">
                    <span className={labelClass}>Bitiş&nbsp;(m)</span>
                    <input
                      type="number"
                      min={0}
                      max={length}
                      step={0.1}
                      className={fieldClasses}
                      value={load.end}
                      onChange={(event) => onUdlChange(load.id, "end", Number(event.target.value))}
                    />
                  </label>
                  <label className="space-y-1">
                    <span className={labelClass}>Yön</span>
                    <select
                      className={fieldClasses}
                      value={load.direction}
                      onChange={(event) => onUdlChange(load.id, "direction", event.target.value)}
                    >
                      <option value="down">Aşağı</option>
                      <option value="up">Yukarı</option>
                    </select>
                  </label>
                  <label className="space-y-1">
                    <span className={labelClass}>Profil</span>
                    <select
                      className={fieldClasses}
                      value={load.shape}
                      onChange={(event) => onUdlChange(load.id, "shape", event.target.value)}
                    >
                      <option value="uniform">Düzgün</option>
                      <option value="triangular_increasing">Üçgen (0 → max)</option>
                      <option value="triangular_decreasing">Üçgen (max → 0)</option>
                    </select>
                  </label>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="space-y-3">
        <header className="flex items-center justify-between">
          <p className={sectionTitleClass}>05 / Moment yükleri <HelpHint label="Moment yükü">Noktasal kuvvet çifti. Büyüklük kN·m cinsindedir. ↺ saat yönü tersi, ↻ saat yönüdür. Pozitif sehim aşağı yönlüdür.</HelpHint></p>
          <button
            type="button"
            onClick={onAddMoment}
            className="rounded-full bg-emerald-500/20 px-3 py-1 text-xs font-semibold text-emerald-200 transition hover:bg-emerald-500/40"
          >
            Moment ekle
          </button>
        </header>
        {momentLoads.length === 0 ? (
          <div className="panel-muted p-4 text-sm text-slate-400">Çerçeve etkilerini simüle etmek için konsantre momentler ekleyin.</div>
        ) : (
          <div className="space-y-3">
            {momentLoads.map((moment) => (
              <div key={moment.id} className="panel-muted relative flex flex-col gap-3 border border-slate-800/60 p-4">
                <button
                  type="button"
                  onClick={() => onRemoveMoment(moment.id)}
                  className="absolute right-2 top-2 flex h-6 w-6 items-center justify-center rounded-full bg-rose-500/20 text-rose-300 transition hover:bg-rose-500/40 hover:text-rose-200"
                  title="Kaldır"
                >
                  ×
                </button>
                <div className="grid gap-3 sm:grid-cols-2">
                  <label className="space-y-1">
                    <span className={labelClass}>Etiket</span>
                    <input
                      type="text"
                      className={fieldClasses}
                      value={moment.id}
                      onChange={(event) => onMomentChange(moment.id, "id", event.target.value.toUpperCase())}
                    />
                  </label>
                  <label className="space-y-1">
                    <span className={labelClass}>Büyüklük (kN·m)</span>
                    <input
                      type="number"
                      min={0}
                      step={0.1}
                      className={fieldClasses}
                      value={moment.magnitude}
                      onChange={(event) => onMomentChange(moment.id, "magnitude", Number(event.target.value))}
                    />
                  </label>
                </div>
                <div className="grid gap-3 sm:grid-cols-2">
                  <label className="space-y-1">
                    <span className={labelClass}>Konum (m)</span>
                    <input
                      type="number"
                      min={0}
                      max={length}
                      step={0.1}
                      className={fieldClasses}
                      value={moment.position}
                      onChange={(event) => onMomentChange(moment.id, "position", Number(event.target.value))}
                    />
                  </label>
                  <label className="space-y-1">
                    <span className={labelClass}>Yön</span>
                    <select
                      className={fieldClasses}
                      value={moment.direction}
                      onChange={(event) => onMomentChange(moment.id, "direction", event.target.value)}
                    >
                      <option value="ccw">Saat yönü tersine</option>
                      <option value="cw">Saat yönünde</option>
                    </select>
                  </label>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      {disableSolveReason && (
        <div className="panel-muted border border-amber-400/30 bg-amber-500/10 p-4 text-xs text-amber-200">
          {disableSolveReason}
        </div>
      )}
    </div>
  );
}


