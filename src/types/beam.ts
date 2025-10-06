export type SupportType = "pin" | "roller";
export type Direction = "down" | "up";
export type MomentDirection = "ccw" | "cw";
export type UdlShape = "uniform" | "triangular_increasing" | "triangular_decreasing";

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
  supports: Array<{ id: string; type: SupportType; position: number }>;
  point_loads: Array<{ id: string; magnitude: number; position: number; angle_deg: number }>;
  udls: Array<{ id: string; magnitude: number; start: number; end: number; direction: Direction; shape: UdlShape }>;
  moment_loads: Array<{ id: string; magnitude: number; position: number; direction: MomentDirection }>;
}

export interface SupportReaction {
  support_id: string;
  support_type: SupportType;
  position: number;
  vertical: number;
  axial: number;
}

export interface BeamSectionHighlight {
  start: number;
  end: number;
  label?: string;
}

export interface BeamSupportInfo {
  id: string;
  type: SupportType;
  position: number;
}

export interface BeamPointLoadInfo {
  id: string;
  magnitude: number;
  position: number;
  angle_deg: number;
}

export interface BeamDistributedLoadInfo {
  id: string;
  magnitude: number;
  start: number;
  end: number;
  direction: Direction;
  shape: UdlShape;
}

export interface BeamMomentLoadInfo {
  id: string;
  magnitude: number;
  position: number;
  direction: MomentDirection;
}

export interface BeamContext {
  length: number;
  supports: BeamSupportInfo[];
  point_loads: BeamPointLoadInfo[];
  udls: BeamDistributedLoadInfo[];
  moment_loads: BeamMomentLoadInfo[];
}

export type AreaTrend = "increase" | "decrease" | "constant";

export interface ShearRegionSamples {
  x: number[];
  shear: number[];
}

export interface MomentSegmentSamples {
  x: number[];
  moment: number[];
}

export interface AreaMethodVisualization {
  shape: string;
  area_value: number;
  trend: AreaTrend;
  region: ShearRegionSamples;
  moment_segment: MomentSegmentSamples;
}

export interface SolutionStep {
  step_number: number;
  title: string;
  explanation: string;
  general_formula?: string;
  substituted_formula?: string;
  numerical_result?: string;
  beam_section?: BeamSectionHighlight;
  area_visualization?: AreaMethodVisualization;
}

export interface SolutionMethod {
  method_name: string;
  method_title: string;
  description: string;
  recommended?: boolean;
  recommendation_reason?: string;
  steps: SolutionStep[];
}

export interface DetailedSolution {
  methods: SolutionMethod[];
  diagram?: {
    x: number[];
    shear: number[];
    moment: number[];
    normal: number[];
  };
  beam_context?: BeamContext;
}

export interface MethodRecommendation {
  method: "shear" | "area";
  title: string;
  reason: string;
}

export interface BeamSolveResponse {
  reactions: SupportReaction[];
  diagram: {
    x: number[];
    shear: number[];
    moment: number[];
    normal: number[];
  };
  derivations: string[];
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
  };
  detailed_solutions?: DetailedSolution;
}
