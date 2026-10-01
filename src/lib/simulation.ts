import type { BeamSimulationResponse } from "../types/beam";

/** Zero-initial-condition response to a sustained step load, all FE modes. */
export function stepDeflection(model: BeamSimulationResponse, time: number): number[] {
  const result = new Array<number>(model.x.length).fill(0);
  const zeta = model.damping_ratio;
  const beta = Math.sqrt(1 - zeta*zeta);
  for (let mode = 0; mode < model.angular_frequencies_rad_s.length; mode++) {
    const omega = model.angular_frequencies_rad_s[mode];
    const phase = omega*beta*time;
    const factor = 1 - Math.exp(-zeta*omega*time)*(Math.cos(phase)+zeta/beta*Math.sin(phase));
    const shape = model.modal_static_shapes_mm[mode];
    for (let j = 0; j < result.length; j++) result[j] += shape[j]*factor;
  }
  return result;
}

export function interpolate(x: number[], y: number[], coordinate: number): number {
  if (!x.length) return 0;
  let i = 0;
  while (i+1 < x.length && x[i+1] < coordinate) i++;
  if (i+1 === x.length || x[i+1] === x[i]) return y[i] || 0;
  return y[i]+(y[i+1]-y[i])*(coordinate-x[i])/(x[i+1]-x[i]);
}
