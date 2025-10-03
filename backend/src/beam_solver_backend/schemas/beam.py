from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, ValidationInfo, model_validator

SupportType = Literal["pin", "roller"]
Direction = Literal["down", "up"]
MomentDirection = Literal["ccw", "cw"]


class Support(BaseModel):
    id: str = Field(min_length=1)
    type: SupportType
    position: float = Field(ge=0)


class PointLoad(BaseModel):
    id: str = Field(min_length=1)
    magnitude: float = Field(gt=0, description="Load magnitude in kN")
    position: float = Field(ge=0, description="Distance from the left end in metres")
    angle_deg: float = Field(default=-90.0, description="Angle measured from +x axis (degrees)")


class UniformDistributedLoad(BaseModel):
    id: str = Field(min_length=1)
    magnitude: float = Field(ge=0, description="Load intensity in kN/m")
    start: float = Field(ge=0, description="Start position in metres")
    end: float = Field(ge=0, description="End position in metres")
    direction: Direction = Field(default="down")


class MomentLoad(BaseModel):
    id: str = Field(min_length=1)
    magnitude: float = Field(gt=0, description="Moment magnitude in kN*m")
    position: float = Field(ge=0, description="Application position in metres")
    direction: MomentDirection = Field(default="ccw")


class Sampling(BaseModel):
    points: int = Field(default=201, ge=51, le=801)


class SolveRequest(BaseModel):
    length: float = Field(gt=0.5, le=30.0)
    supports: List[Support] = Field(default_factory=list)
    point_loads: List[PointLoad] = Field(default_factory=list)
    udls: List[UniformDistributedLoad] = Field(default_factory=list)
    moment_loads: List[MomentLoad] = Field(default_factory=list)
    sampling: Optional[Sampling] = None

    @model_validator(mode="after")
    def validate_domain(self, info: ValidationInfo) -> "SolveRequest":
        if len(self.supports) != 2:
            raise ValueError("Exactly two supports are required for the current solver.")

        length = self.length
        positions = sorted(support.position for support in self.supports)
        if positions[0] < 0 or positions[1] > length:
            raise ValueError("Support positions must lie within the beam span.")
        if abs(positions[0] - positions[1]) < 1e-6:
            raise ValueError("Support positions must be distinct.")

        for load in self.point_loads:
            if not 0 <= load.position <= length:
                raise ValueError("Point load must be located on the beam span.")

        for udl in self.udls:
            if not (0 <= udl.start < udl.end <= length):
                raise ValueError("UDL start and end must be ordered within the beam span.")

        for moment in self.moment_loads:
            if not 0 <= moment.position <= length:
                raise ValueError("Moment application position must lie on the beam span.")

        if self.sampling and self.sampling.points % 2 == 0:
            # Odd number of points gives symmetric sampling for trapezoidal integration.
            pass

        return self


class SupportReaction(BaseModel):
    support_id: str
    support_type: SupportType
    position: float
    vertical: float
    axial: float


class DiagramData(BaseModel):
    x: List[float]
    shear: List[float]
    moment: List[float]
    normal: List[float]


class SolveMeta(BaseModel):
    solve_time_ms: float
    validation_warnings: List[str]


class SolutionStep(BaseModel):
    step_number: int
    title: str
    explanation: str
    general_formula: Optional[str] = None  # Genel formül (LaTeX)
    substituted_formula: Optional[str] = None  # Değerlerin yerine konulmuş hali (LaTeX)
    numerical_result: Optional[str] = None


class SolutionMethod(BaseModel):
    method_name: str
    method_title: str
    description: str
    steps: List[SolutionStep]


class DetailedSolution(BaseModel):
    methods: List[SolutionMethod]
    diagram: Optional[DiagramData] = None  # Grafik verileri çözüm adımlarında gösterilmek üzere


class SolveResponse(BaseModel):
    reactions: List[SupportReaction]
    diagram: DiagramData
    derivations: List[str]
    meta: SolveMeta
    detailed_solutions: Optional[DetailedSolution] = None
