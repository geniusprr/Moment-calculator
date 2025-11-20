from __future__ import annotations

import math
from typing import List

from beam_solver_backend.schemas.chimney import ChimneyPeriodRequest, ChimneyPeriodResponse


BETA1 = 1.875104068711961  # Cantilever 1. mod şekil katsayısı


def calculate_fundamental_period(payload: ChimneyPeriodRequest) -> ChimneyPeriodResponse:
    height = payload.height_m
    ei = payload.elastic_modulus_gpa * 1e9 * payload.moment_inertia_m4  # N·m²
    m_line = payload.mass_per_length_kgm
    tip_equivalent = payload.tip_mass_kg / height if payload.tip_mass_kg > 0 else 0.0
    m_effective = m_line + tip_equivalent

    omega = (BETA1**2) * math.sqrt(ei / (m_effective * (height**4)))
    period = 2 * math.pi / omega
    frequency = 1.0 / period

    notes: List[str] = [
        "Model: Tekil ankastre uçlu, süreklı kütle yayılı baca.",
        "Formül: ω₁ = β₁²·√(EI / (m·H⁴)), T₁ = 2π/ω₁",
        f"β₁ = {BETA1:.4f} (1. mod cantilever)",
    ]
    if payload.tip_mass_kg > 0:
        notes.append("Serbest uç ek kütlesi, eşdeğer yayılı kütle olarak H ile bölünüp m'a eklendi.")

    return ChimneyPeriodResponse(
        period_s=period,
        frequency_hz=frequency,
        angular_frequency_rad_s=omega,
        flexural_rigidity_n_m2=ei,
        effective_mass_kgm=m_effective,
        mode_constant=BETA1,
        notes=notes,
    )
