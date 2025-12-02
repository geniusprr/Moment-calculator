from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, ValidationInfo, model_validator

SupportType = Literal["pin", "roller", "fixed"]
Direction = Literal["down", "up"]
MomentDirection = Literal["ccw", "cw"]
DistributedLoadShape = Literal["uniform", "triangular_increasing", "triangular_decreasing"]
BeamType = Literal["simply_supported", "cantilever"]


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
        """Ensure the provided loads and supports form a valid beam model."""
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


class SolveResponse(BaseModel):
    reactions: List[SupportReaction]
    diagram: DiagramData
    meta: SolveMeta


class ChimneyPeriodRequest(BaseModel):
    height_m: float = Field(gt=1.0, description="Baca yüksekliği (m)")
    elastic_modulus_gpa: float = Field(gt=0.0, description="Elastisite modülü (GPa)")
    moment_inertia_m4: float = Field(gt=0.0, description="Kesit atalet momenti (m^4)")
    mass_per_length_kgm: float = Field(gt=0.0, description="Doğrusal kütle (kg/m)")
    tip_mass_kg: float = Field(default=0.0, ge=0.0, description="Serbest uçta ek kütle (kg)")


class ChimneyPeriodResponse(BaseModel):
    period_s: float
    frequency_hz: float
    angular_frequency_rad_s: float
    flexural_rigidity_n_m2: float
    effective_mass_kgm: float
    mode_constant: float
    notes: List[str]
