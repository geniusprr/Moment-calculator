from __future__ import annotations

from typing import List, Literal, Optional, Any

from pydantic import BaseModel, Field, ValidationInfo, model_validator

SupportType = Literal["pin", "roller", "fixed"]
Direction = Literal["down", "up"]
MomentDirection = Literal["ccw", "cw"]
DistributedLoadShape = Literal["uniform", "triangular_increasing", "triangular_decreasing"]
BeamType = Literal["simply_supported", "cantilever"]
AreaTrend = Literal["increase", "decrease", "constant"]


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
    shape: DistributedLoadShape = Field(default="uniform")


class MomentLoad(BaseModel):
    id: str = Field(min_length=1)
    magnitude: float = Field(gt=0, description="Moment magnitude in kN*m")
    position: float = Field(ge=0, description="Application position in metres")
    direction: MomentDirection = Field(default="ccw")


class SolveRequest(BaseModel):
    length: float = Field(gt=0.5, le=30.0)
    supports: List[Support] = Field(default_factory=list)
    point_loads: List[PointLoad] = Field(default_factory=list)
    udls: List[UniformDistributedLoad] = Field(default_factory=list)
    moment_loads: List[MomentLoad] = Field(default_factory=list)
    beam_type: BeamType = Field(default="simply_supported", description="Beam boundary condition configuration")

    @model_validator(mode="after")
    def validate_domain(self, info: ValidationInfo) -> "SolveRequest":
        length = self.length

        if self.beam_type == "simply_supported":
            if len(self.supports) != 2:
                raise ValueError("Basit kiris icin tam olarak iki mesnet tanimlanmalidir.")

            positions = sorted(support.position for support in self.supports)
            if positions[0] < 0 or positions[1] > length:
                raise ValueError("Mesnet konumlari kiris boyu icinde olmalidir.")
            if abs(positions[0] - positions[1]) < 1e-6:
                raise ValueError("Mesnet konumlari ayri olmalidir.")

            invalid_fixed = [support for support in self.supports if support.type == "fixed"]
            if invalid_fixed:
                raise ValueError("Basit kiris seceneginde ankastre (fixed) mesnet kullanilamaz.")
        else:
            if len(self.supports) != 1:
                raise ValueError("Konsol kiris icin tek bir ankastre mesnet gereklidir.")

            only_support = self.supports[0]
            if only_support.type != "fixed":
                raise ValueError("Konsol kiris icin mesnet tipi fixed olmalidir.")

            if not (abs(only_support.position) < 1e-9 or abs(only_support.position - length) < 1e-9):
                raise ValueError("Konsol mesneti kirisin baslangicinda veya ucunda olmalidir (x=0 veya x=L).")

        for load in self.point_loads:
            if not 0 <= load.position <= length:
                raise ValueError("Point load must be located on the beam span.")

        for udl in self.udls:
            if not (0 <= udl.start < udl.end <= length):
                raise ValueError("UDL start and end must be ordered within the beam span.")
            if udl.magnitude <= 0:
                raise ValueError("Distributed load magnitude must be positive.")

        for moment in self.moment_loads:
            if not 0 <= moment.position <= length:
                raise ValueError("Moment application position must lie on the beam span.")

        return self


class SupportReaction(BaseModel):
    support_id: str
    support_type: SupportType
    position: float
    vertical: float
    axial: float
    moment: float = Field(default=0.0, description="Support fixing moment for cantilever cases (kN*m)")


class DiagramData(BaseModel):
    x: List[float]
    shear: List[float]
    moment: List[float]
    normal: List[float]


class MethodRecommendation(BaseModel):
    method: Literal["shear", "area"]
    title: str
    reason: str


class SolveMeta(BaseModel):
    solve_time_ms: float
    validation_warnings: List[str]
    recommendation: Optional[MethodRecommendation] = None
    max_positive_moment: Optional[float] = None
    max_positive_position: Optional[float] = None
    min_negative_moment: Optional[float] = None
    min_negative_position: Optional[float] = None
    max_absolute_moment: Optional[float] = None
    max_absolute_position: Optional[float] = None


# --- Detailed Solution Types ---

class BeamSectionHighlight(BaseModel):
    start: float
    end: float
    label: Optional[str] = None


class ShearRegionSamples(BaseModel):
    x: List[float]
    shear: List[float]


class MomentSegmentSamples(BaseModel):
    x: List[float]
    moment: List[float]


class AreaMethodVisualization(BaseModel):
    shape: str
    area_value: float
    trend: AreaTrend
    region: ShearRegionSamples
    moment_segment: MomentSegmentSamples


class SolutionStep(BaseModel):
    step_number: int
    title: str
    explanation: str
    general_formula: Optional[str] = None
    substituted_formula: Optional[str] = None
    numerical_result: Optional[str] = None
    beam_section: Optional[BeamSectionHighlight] = None
    area_visualization: Optional[AreaMethodVisualization] = None


class SolutionMethod(BaseModel):
    method_name: str
    method_title: str
    description: str
    recommended: bool = False
    recommendation_reason: Optional[str] = None
    steps: List[SolutionStep]


class BeamContext(BaseModel):
    length: float
    supports: List[Support]
    point_loads: List[PointLoad]
    udls: List[UniformDistributedLoad]
    moment_loads: List[MomentLoad]


class DetailedSolution(BaseModel):
    methods: List[SolutionMethod]
    diagram: Optional[DiagramData] = None
    beam_context: Optional[BeamContext] = None


class SolveResponse(BaseModel):
    reactions: List[SupportReaction]
    diagram: DiagramData
    derivations: List[str]
    meta: SolveMeta
    detailed_solutions: Optional[DetailedSolution] = None

