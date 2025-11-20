from __future__ import annotations

from pydantic import BaseModel, Field


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
    notes: list[str]
