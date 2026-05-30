export type SupportType = "pin" | "roller" | "fixed";
export type Direction = "down" | "up";
export type MomentDirection = "ccw" | "cw";
export type UdlShape = "uniform" | "triangular_increasing" | "triangular_decreasing";
export type BeamType = "simply_supported" | "cantilever";

export interface SupportInput {
  id: string;
  type: SupportType;
  position: number;
}

export interface PointLoadInput {
  id: string;
  magnitude: number;
  position: number;
  angleDeg: number;
}

export interface UdlInput {
  id: string;
  magnitude: number;
  start: number;
  end: number;
  direction: Direction;
  shape: UdlShape;
}

export interface MomentLoadInput {
  id: string;
  magnitude: number;
  position: number;
  direction: MomentDirection;
}

export interface BeamSolveRequest {
  length: number;
  beam_type: BeamType;
  supports: Array<{ id: string; type: SupportType; position: number }>;
  point_loads: Array<{ id: string; magnitude: number; position: number; angle_deg: number }>;
  udls: Array<{ id: string; magnitude: number; start: number; end: number; direction: Direction; shape: UdlShape }>;
  moment_loads: Array<{ id: string; magnitude: number; position: number; direction: MomentDirection }>;
  elastic_modulus_gpa?: number;
  moment_inertia_m4?: number;
}

export interface SupportReaction {
  support_id: string;
  support_type: SupportType;
  position: number;
  vertical: number;
  axial: number;
  moment?: number;
}

export interface LoadColorConfig {
  point: string;
  uniformUdl: string;
  triangularUdl: string;
  moment: string;
}

export interface MethodRecommendation {
  method: "shear" | "area" | "integration";
  title: string;
  reason: string;
}

export interface BeamContext {
  length: number;
  supports: SupportInput[];
  point_loads: Array<{ id: string; magnitude: number; position: number; angle_deg: number }>;
  udls: UdlInput[];
  moment_loads: MomentLoadInput[];
}

export interface BeamSectionHighlight {
  start: number;
  end: number;
  label?: string | null;
}

export interface AreaMethodVisualization {
  region: {
    x: number[];
    shear: number[];
  };
  moment_segment: {
    x: number[];
    moment: number[];
  };
  shape: string;
  area_value: number;
  trend: "increase" | "decrease" | "constant";
}

export interface SolutionStep {
  step_number: number;
  title: string;
  explanation: string;
  general_formula?: string | null;
  substituted_formula?: string | null;
  numerical_result?: string | null;
  beam_section?: BeamSectionHighlight | null;
  area_visualization?: AreaMethodVisualization | null;
}

export interface SolutionMethod {
  method_name: string;
  method_title: string;
  description: string;
  recommended?: boolean;
  recommendation_reason?: string | null;
  steps: SolutionStep[];
}

export interface DetailedSolution {
  methods: SolutionMethod[];
  beam_context?: BeamContext | null;
  diagram?: BeamSolveResponse["diagram"] | null;
}

// Chimney dynamic period
export interface ChimneyPeriodRequest {
  height_m: number;
  elastic_modulus_gpa: number;
  moment_inertia_m4: number;
  mass_per_length_kgm: number;
  tip_mass_kg?: number;
}

export interface ChimneyPeriodResponse {
  period_s: number;
  frequency_hz: number;
  angular_frequency_rad_s: number;
  flexural_rigidity_n_m2: number;
  effective_mass_kgm: number;
  mode_constant: number;
  notes: string[];
}

export interface BeamSolveResponse {
  reactions: SupportReaction[];
  diagram: {
    x: number[];
    shear: number[];
    moment: number[];
    normal: number[];
    deflection?: number[];
    rotation?: number[];
  };
  meta: {
    solve_time_ms: number;
    validation_warnings: string[];
    recommendation: MethodRecommendation;
    max_positive_moment?: number;
    max_positive_position?: number;
    min_negative_moment?: number;
    min_negative_position?: number;
    max_absolute_moment?: number;
    max_absolute_position?: number;
    max_deflection?: number;
    max_deflection_position?: number;
  };
  detailed_solution?: DetailedSolution;
}
