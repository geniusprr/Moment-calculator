import type { BeamSimulationResponse, BeamSolveRequest, BeamSolveResponse, ChimneyPeriodRequest, ChimneyPeriodResponse } from "@/types/beam";

const baseUrl = (process.env.NEXT_PUBLIC_API_BASE_URL || "/api").replace(/\/$/, "");

async function post<T>(path: string, payload: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${baseUrl}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = body?.detail;
    const message = typeof detail === "string" ? detail : Array.isArray(detail)
      ? detail.map((item: { loc?: Array<string | number>; msg?: string }) => `${item.loc?.slice(1).join(" → ")}: ${item.msg}`).join("; ")
      : "Hesaplama servisine ulaşılamadı. Lütfen tekrar deneyin.";
    throw new Error(message);
  }
  return body as T;
}

export const solveBeam = (payload: BeamSolveRequest, signal?: AbortSignal) => post<BeamSolveResponse>("/solve", payload, signal);
export const solveChimneyPeriod = (payload: ChimneyPeriodRequest) => post<ChimneyPeriodResponse>("/chimney/period", payload);
export const simulateBeam = (payload: BeamSolveRequest & { mass_per_length_kgm: number; damping_ratio: number }, signal?: AbortSignal) => post<BeamSimulationResponse>("/simulate", payload, signal);
