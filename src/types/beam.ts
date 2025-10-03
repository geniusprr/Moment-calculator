export type SupportType = "pin" | "roller";
export type Direction = "down" | "up";
export type MomentDirection = "ccw" | "cw";

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
  udls: Array<{ id: string; magnitude: number; start: number; end: number; direction: Direction }>;
  moment_loads: Array<{ id: string; magnitude: number; position: number; direction: MomentDirection }>;
  sampling?: {
    points: number;
  };
}

export interface SupportReaction {
  support_id: string;
  support_type: SupportType;
  position: number;
  vertical: number;
  axial: number;
}

export interface SolutionStep {
  step_number: number;
  title: string;
  explanation: string;
  general_formula?: string;
  substituted_formula?: string;
  numerical_result?: string;
}

export interface SolutionMethod {
  method_name: string;
  method_title: string;
  description: string;
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
  };
  detailed_solutions?: DetailedSolution;
}
