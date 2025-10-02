"use client";

import type {
  MomentLoadInput,
  PointLoadInput,
  SupportInput,
  UdlInput,
} from "@/types/beam";

interface BeamFormProps {
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
  samplingPoints: number;
  onSamplingChange: (value: number) => void;
  onSolve: () => void;
  onReset: () => void;
  solving: boolean;
  disableSolveReason?: string | null;
}

const fieldClasses =
  "w-full rounded-xl border border-slate-700/80 bg-slate-900/80 px-4 py-2 text-sm text-slate-100 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-500/40";

const sectionTitleClass = "text-sm font-semibold text-slate-200";
const sectionHintClass = "text-xs text-slate-500";
const labelClass = "text-xs font-medium uppercase tracking-wide text-slate-400";

export function BeamForm({
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
  samplingPoints,
  onSamplingChange,
  onSolve,
  onReset,
  solving,
  disableSolveReason,
}: BeamFormProps) {
  return (
    <div className="panel space-y-6 p-6">
      <div className="flex items-center justify-between">
        <div>
          <span className="tag">Model Girişleri</span>
          <p className="text-sm text-slate-400">Kiriş uzunluğu, mesnetler ve yük durumları</p>
        </div>
        <button
          onClick={onReset}
          className="rounded-full border border-slate-700/70 px-3 py-1 text-xs text-slate-300 transition hover:border-slate-500 hover:text-white"
          type="button"
        >
          Sıfırla
        </button>
      </div>

      <section className="grid gap-4 sm:grid-cols-2">
        <label className="space-y-2">
          <span className={labelClass}>Kiriş uzunluğu (m)</span>
          <input
            type="number"
            min={0.5}
            max={20}
            step={0.1}
            className={fieldClasses}
            value={length}
            onChange={(event) => onLengthChange(Number(event.target.value))}
          />
        </label>
        <label className="space-y-2">
          <span className={labelClass}>Örnekleme noktaları</span>
          <input
            type="number"
            min={101}
            max={801}
            step={2}
            className={fieldClasses}
            value={samplingPoints}
            onChange={(event) => onSamplingChange(Number(event.target.value))}
          />
        </label>
      </section>

      <section className="space-y-3">
        <header className="flex items-center justify-between">
          <div>
            <p className={sectionTitleClass}>Mesnetler</p>
            <p className={sectionHintClass}>Statik çözüm için tam olarak iki mesnet gereklidir.</p>
          </div>
          <button
            type="button"
            onClick={onAddSupport}
            disabled={supports.length >= 2}
            className="rounded-full bg-slate-800/70 px-3 py-1 text-xs font-semibold text-slate-200 transition hover:bg-slate-700/80 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Mesnet ekle
          </button>
        </header>
        {supports.length === 0 ? (
          <div className="panel-muted p-4 text-sm text-slate-400">Çözümü etkinleştirmek için iki mesnet ekleyin.</div>
        ) : (
          <div className="space-y-3">
            {supports.map((support) => (
              <div key={support.id} className="panel-muted grid gap-3 border border-slate-800/60 p-4 sm:grid-cols-5">
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
                  >
                    <option value="pin">Menteşe</option>
                    <option value="roller">Kayar</option>
                  </select>
                </label>
                <label className="space-y-1 sm:col-span-2">
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
                <div className="flex items-end justify-end">
                  <button
                    type="button"
                    onClick={() => onRemoveSupport(support.id)}
                    className="text-xs text-rose-300 transition hover:text-rose-200"
                  >
                    Kaldır
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="space-y-3">
        <header className="flex items-center justify-between">
          <p className={sectionTitleClass}>Tekil yükler</p>
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
              <div key={load.id} className="panel-muted grid gap-3 border border-slate-800/60 p-4 sm:grid-cols-6">
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
                <label className="space-y-1 sm:col-span-2">
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
                <div className="flex items-end justify-end">
                  <button
                    type="button"
                    onClick={() => onRemovePointLoad(load.id)}
                    className="text-xs text-rose-300 transition hover:text-rose-200"
                  >
                    Kaldır
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="space-y-3">
        <header className="flex items-center justify-between">
          <p className={sectionTitleClass}>Yayılı yükler</p>
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
              <div key={load.id} className="panel-muted grid gap-3 border border-slate-800/60 p-4 sm:grid-cols-6">
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
                <label className="space-y-1">
                  <span className={labelClass}>Başlangıç (m)</span>
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
                  <span className={labelClass}>Bitiş (m)</span>
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
                <div className="flex items-end justify-end">
                  <button
                    type="button"
                    onClick={() => onRemoveUdl(load.id)}
                    className="text-xs text-rose-300 transition hover:text-rose-200"
                  >
                    Kaldır
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="space-y-3">
        <header className="flex items-center justify-between">
          <p className={sectionTitleClass}>Tekil momentler</p>
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
              <div key={moment.id} className="panel-muted grid gap-3 border border-slate-800/60 p-4 sm:grid-cols-5">
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
                <div className="flex items-end justify-end">
                  <button
                    type="button"
                    onClick={() => onRemoveMoment(moment.id)}
                    className="text-xs text-rose-300 transition hover:text-rose-200"
                  >
                    Kaldır
                  </button>
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

      <div className="flex flex-col-reverse gap-3 sm:flex-row sm:justify-end">
        <button
          type="button"
          onClick={onSolve}
          disabled={solving || Boolean(disableSolveReason)}
          className="group relative overflow-hidden rounded-full bg-cyan-500 px-6 py-2 text-sm font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:bg-slate-700/60 disabled:text-slate-400"
        >
          <span className="relative z-10 flex items-center gap-2">
            {solving && <span className="inline-flex h-3 w-3 animate-ping rounded-full bg-slate-950 opacity-80" />}
            {solving ? "Çözülüyor" : "Çöz"}
          </span>
          <span className="absolute inset-0 -z-0 translate-y-full bg-white/40 transition group-hover:translate-y-0" />
        </button>
      </div>
    </div>
  );
}


