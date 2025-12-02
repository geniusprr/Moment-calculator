from __future__ import annotations

import math
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from beam_solver_backend.schemas import (
    ChimneyPeriodRequest,
    ChimneyPeriodResponse,
    DiagramData,
    MethodRecommendation,
    SolveMeta,
    SolveRequest,
    SolveResponse,
    SupportReaction,
)

DEFAULT_SAMPLING_POINTS = 401
ROOT_TOL = 1e-9
MomentCandidate = Tuple[float, float]
BETA1 = 1.875104068711961  # Cantilever 1. mode shape constant


def _format_float(value: float) -> float:
    """Format numeric values to six decimals for deterministic JSON output."""
    return float(f"{value:.6f}")


def _vertical_component(load) -> float:
    """Return the downward vertical component of an angled point load."""
    angle_rad = math.radians(load.angle_deg)
    vertical = -load.magnitude * math.sin(angle_rad)
    if abs(vertical) < 1e-9:
        return 0.0
    return vertical


def _axial_component(load) -> float:
    """Return the horizontal component of an angled point load."""
    angle_rad = math.radians(load.angle_deg)
    axial = load.magnitude * math.cos(angle_rad)
    if abs(axial) < 1e-9:
        return 0.0
    return axial


def _udl_sign(udl) -> float:
    """Return +1/-1 based on whether the distributed load acts downward or upward."""
    return 1.0 if udl.direction == "down" else -1.0


def _udl_equivalent_force_and_centroid(udl) -> tuple[float, float]:
    """Convert a distributed load segment into an equivalent point force and centroid."""
    span = udl.end - udl.start
    if span <= 0:
        raise ValueError("Distributed load span must be positive.")

    sign = _udl_sign(udl)

    if udl.shape == "uniform":
        equivalent_force = sign * udl.magnitude * span
        centroid = udl.start + span / 2.0
    elif udl.shape == "triangular_increasing":
        equivalent_force = sign * 0.5 * udl.magnitude * span
        centroid = udl.start + (2.0 * span) / 3.0
    elif udl.shape == "triangular_decreasing":
        equivalent_force = sign * 0.5 * udl.magnitude * span
        centroid = udl.start + span / 3.0
    else:
        raise ValueError(f"Unsupported distributed load shape: {udl.shape}")

    return equivalent_force, centroid


def _udl_shear_contribution(udl, x_axis: np.ndarray) -> np.ndarray:
    """Compute the shear contribution of a distributed load across the axis."""
    span = udl.end - udl.start
    if span <= 0:
        return np.zeros_like(x_axis, dtype=float)

    xi = np.clip(x_axis - udl.start, 0.0, span)
    sign = _udl_sign(udl)

    if udl.shape == "uniform":
        return sign * udl.magnitude * xi
    if udl.shape == "triangular_increasing":
        return sign * (udl.magnitude * xi**2) / (2.0 * span)
    if udl.shape == "triangular_decreasing":
        return sign * (udl.magnitude * (xi - (xi**2) / (2.0 * span)))

    raise ValueError(f"Unsupported distributed load shape: {udl.shape}")


def _udl_moment_contribution(udl, x_axis: np.ndarray) -> np.ndarray:
    """Compute the bending moment contribution of a distributed load."""
    span = udl.end - udl.start
    if span <= 0:
        return np.zeros_like(x_axis, dtype=float)

    base = np.maximum(x_axis - udl.start, 0.0)
    xi = np.clip(base, 0.0, span)
    w = _udl_sign(udl) * udl.magnitude

    if udl.shape == "uniform":
        return w * ((base * xi) - 0.5 * xi**2)

    if udl.shape == "triangular_increasing":
        return (w / span) * ((base * (xi**2)) / 2.0 - (xi**3) / 3.0)

    if udl.shape == "triangular_decreasing":
        return w * (
            base * xi
            - 0.5 * xi**2
            - (base * (xi**2)) / (2.0 * span)
            + (xi**3) / (3.0 * span)
        )

    raise ValueError(f"Unsupported distributed load shape: {udl.shape}")


def _add_unique_point(points: List[float], value: float, beam_length: float, tol: float = 1e-9) -> None:
    """Append a location to the critical point list if it is new within tolerance."""
    if math.isnan(value) or math.isinf(value):
        return
    clamped = min(max(value, 0.0), beam_length)
    for existing in points:
        if math.isclose(existing, clamped, abs_tol=tol, rel_tol=0.0):
            return
    points.append(clamped)


def _moment_value(payload: SolveRequest, reactions: List[SupportReaction], position: float) -> float:
    """Evaluate the bending moment diagram at the requested coordinate."""
    clamped = min(max(position, 0.0), payload.length)
    return float(_moment_diagram(payload, np.array([clamped], dtype=float), reactions)[0])


def _register_moment_candidate(
    candidates: List[MomentCandidate],
    payload: SolveRequest,
    reactions: List[SupportReaction],
    position: float,
    tol: float = 1e-6,
) -> None:
    """Register a candidate coordinate/value pair for moment extrema detection."""
    if math.isnan(position) or math.isinf(position):
        return
    clamped = min(max(position, 0.0), payload.length)
    for existing_x, _ in candidates:
        if math.isclose(existing_x, clamped, abs_tol=tol, rel_tol=0.0):
            return
    candidates.append((clamped, _moment_value(payload, reactions, clamped)))


def _compute_moment_extrema(
    payload: SolveRequest,
    reactions: List[SupportReaction],
    x_axis: np.ndarray,
    shear: np.ndarray,
) -> Dict[str, Optional[MomentCandidate]]:
    """Search the shear diagram for zero crossings to locate key moment values."""
    candidates: List[MomentCandidate] = []

    _register_moment_candidate(candidates, payload, reactions, 0.0)
    _register_moment_candidate(candidates, payload, reactions, payload.length)

    for idx in range(len(x_axis) - 1):
        left = float(x_axis[idx])
        right = float(x_axis[idx + 1])
        s_left = float(shear[idx])
        s_right = float(shear[idx + 1])

        if math.isclose(s_left, 0.0, abs_tol=ROOT_TOL):
            _register_moment_candidate(candidates, payload, reactions, left)
        if math.isclose(s_right, 0.0, abs_tol=ROOT_TOL):
            _register_moment_candidate(candidates, payload, reactions, right)

        sign_change = s_left * s_right < 0.0
        near_zero = (abs(s_left) < 1e-6) or (abs(s_right) < 1e-6)

        if sign_change or near_zero:
            root = _locate_shear_zero(payload, reactions, left, right, s_left, s_right)
            _register_moment_candidate(candidates, payload, reactions, root)
            continue

        mid = 0.5 * (left + right)
        s_mid = float(_shear_diagram(payload, np.array([mid], dtype=float), reactions)[0])
        if abs(s_mid) < 1e-6:
            _register_moment_candidate(candidates, payload, reactions, mid)
        if s_left * s_mid < 0.0:
            root = _locate_shear_zero(payload, reactions, left, mid, s_left, s_mid)
            _register_moment_candidate(candidates, payload, reactions, root)
        elif s_mid * s_right < 0.0:
            root = _locate_shear_zero(payload, reactions, mid, right, s_mid, s_right)
            _register_moment_candidate(candidates, payload, reactions, root)

    if not candidates:
        mid = 0.5 * payload.length
        _register_moment_candidate(candidates, payload, reactions, mid)

    max_positive_candidates = [candidate for candidate in candidates if candidate[1] >= -1e-9]
    max_positive = max(max_positive_candidates, key=lambda item: item[1]) if max_positive_candidates else max(
        candidates, key=lambda item: item[1]
    )

    negative_candidates = [candidate for candidate in candidates if candidate[1] <= -1e-9]
    min_negative = min(negative_candidates, key=lambda item: item[1]) if negative_candidates else None

    max_absolute = max(candidates, key=lambda item: abs(item[1]))

    return {
        "max_positive": max_positive,
        "min_negative": min_negative,
        "max_absolute": max_absolute,
    }


def _locate_shear_zero(
    payload: SolveRequest,
    reactions: List[SupportReaction],
    left: float,
    right: float,
    shear_left: float,
    shear_right: float,
    max_iterations: int = 60,
    tol: float = ROOT_TOL,
) -> float:
    """Use bisection refinements to locate the root of the shear function."""
    if abs(shear_left) < tol:
        return left
    if abs(shear_right) < tol:
        return right

    lo, hi = left, right
    f_lo, f_hi = shear_left, shear_right

    if f_lo * f_hi > 0.0:
        mid = 0.5 * (lo + hi)
        f_mid = float(_shear_diagram(payload, np.array([mid], dtype=float), reactions)[0])
        if f_lo * f_mid <= 0.0:
            return _locate_shear_zero(payload, reactions, lo, mid, f_lo, f_mid, max_iterations, tol)
        if f_mid * f_hi <= 0.0:
            return _locate_shear_zero(payload, reactions, mid, hi, f_mid, f_hi, max_iterations, tol)
        return mid

    for _ in range(max_iterations):
        mid = 0.5 * (lo + hi)
        f_mid = float(_shear_diagram(payload, np.array([mid], dtype=float), reactions)[0])
        if abs(f_mid) < tol or (hi - lo) < tol:
            return mid
        if f_lo * f_mid <= 0.0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid

    return 0.5 * (lo + hi)


def _determine_method_recommendation(payload: SolveRequest) -> MethodRecommendation:
    """Return a simple method recommendation payload for the UI."""
    return MethodRecommendation(
        method="area",
        title="Alan Yontemi",
        reason="Standart cozum yontemi.",
    )


def _moment_sign(direction) -> float:
    """Map textual moment direction to a numerical sign."""
    return 1.0 if direction == "ccw" else -1.0


def _compute_reactions(payload: SolveRequest) -> List[SupportReaction]:
    """Solve statics for a simply supported beam and return support reactions."""
    supports_sorted = sorted(payload.supports, key=lambda support: support.position)
    support_a, support_b = supports_sorted
    span = support_b.position - support_a.position
    if span <= 0:
        raise ValueError("Right support must be placed after the left support.")

    total_vertical = 0.0
    total_moment_about_a = 0.0
    total_axial = 0.0

    for load in payload.point_loads:
        vertical = _vertical_component(load)
        axial = _axial_component(load)
        total_vertical += vertical
        total_axial += axial
        lever_arm = load.position - support_a.position
        total_moment_about_a += vertical * lever_arm

    for udl in payload.udls:
        equivalent_force, centroid = _udl_equivalent_force_and_centroid(udl)
        total_vertical += equivalent_force
        total_moment_about_a += equivalent_force * (centroid - support_a.position)

    for moment in payload.moment_loads:
        total_moment_about_a += moment.magnitude * _moment_sign(moment.direction)

    if math.isclose(span, 0.0):
        raise ValueError("Support span cannot be zero.")

    reaction_b_vertical = total_moment_about_a / span
    reaction_a_vertical = total_vertical - reaction_b_vertical
    reaction_a_axial = -total_axial
    reaction_b_axial = 0.0

    reactions = [
        SupportReaction(
            support_id=support_a.id,
            support_type=support_a.type,
            position=_format_float(support_a.position),
            vertical=_format_float(reaction_a_vertical),
            axial=_format_float(reaction_a_axial),
        ),
        SupportReaction(
            support_id=support_b.id,
            support_type=support_b.type,
            position=_format_float(support_b.position),
            vertical=_format_float(reaction_b_vertical),
            axial=_format_float(reaction_b_axial),
        ),
    ]

    return reactions


def _shear_diagram(payload: SolveRequest, x_axis: np.ndarray, reactions: List[SupportReaction]) -> np.ndarray:
    """Build the shear diagram by superposing reactions, point loads and UDLs."""
    shear = np.zeros_like(x_axis, dtype=float)

    for reaction in reactions:
        shear += reaction.vertical * (x_axis >= reaction.position)

    for load in payload.point_loads:
        vertical = _vertical_component(load)
        shear -= vertical * (x_axis >= load.position)

    for udl in payload.udls:
        shear -= _udl_shear_contribution(udl, x_axis)

    return shear


def _normal_diagram(payload: SolveRequest, x_axis: np.ndarray, reactions: List[SupportReaction]) -> np.ndarray:
    """Compute the axial force diagram using horizontal components of loads."""
    normal = np.zeros_like(x_axis, dtype=float)

    for reaction in reactions:
        normal += reaction.axial * (x_axis >= reaction.position)

    for load in payload.point_loads:
        axial = _axial_component(load)
        normal += axial * (x_axis >= load.position)

    return normal


def _moment_diagram(payload: SolveRequest, x_axis: np.ndarray, reactions: List[SupportReaction]) -> np.ndarray:
    """Integrate shear effects and applied moments to obtain bending moment values."""
    moment = np.zeros_like(x_axis, dtype=float)

    for reaction in reactions:
        offsets = np.maximum(x_axis - reaction.position, 0.0)
        moment += reaction.vertical * offsets
        if hasattr(reaction, "moment"):
            moment += getattr(reaction, "moment", 0.0) * (x_axis >= reaction.position)

    for load in payload.point_loads:
        vertical = _vertical_component(load)
        offsets = np.maximum(x_axis - load.position, 0.0)
        moment -= vertical * offsets

    for udl in payload.udls:
        moment -= _udl_moment_contribution(udl, x_axis)

    for load in payload.moment_loads:
        signed = load.magnitude * _moment_sign(load.direction)
        moment += signed * (x_axis >= load.position)

    return moment


def solve_beam(payload: SolveRequest) -> SolveResponse:
    """Public entry point that returns reactions plus shear/moment/normal diagrams."""
    start_time = perf_counter()
    reactions = _compute_reactions(payload)
    recommendation = _determine_method_recommendation(payload)

    sampling_points = DEFAULT_SAMPLING_POINTS
    base_axis = np.linspace(0.0, payload.length, num=sampling_points, dtype=float, endpoint=True)

    critical_points = base_axis.tolist()
    for support in payload.supports:
        _add_unique_point(critical_points, support.position, payload.length)
    for load in payload.point_loads:
        _add_unique_point(critical_points, load.position, payload.length)
    for udl in payload.udls:
        _add_unique_point(critical_points, udl.start, payload.length)
        _add_unique_point(critical_points, udl.end, payload.length)
    for moment_load in payload.moment_loads:
        _add_unique_point(critical_points, moment_load.position, payload.length)

    x_axis = np.array(sorted(critical_points), dtype=float)

    shear = _shear_diagram(payload, x_axis, reactions)
    moment = _moment_diagram(payload, x_axis, reactions)
    normal = _normal_diagram(payload, x_axis, reactions)

    moment_extrema = _compute_moment_extrema(payload, reactions, x_axis, shear)

    duration_ms = (perf_counter() - start_time) * 1000.0

    max_positive = moment_extrema.get("max_positive")
    min_negative = moment_extrema.get("min_negative")
    max_absolute = moment_extrema.get("max_absolute")

    diagram_data = DiagramData(
        x=[_format_float(v) for v in x_axis],
        shear=[_format_float(v) for v in shear],
        moment=[_format_float(v) for v in moment],
        normal=[_format_float(v) for v in normal],
    )

    return SolveResponse(
        reactions=reactions,
        diagram=diagram_data,
        meta=SolveMeta(
            solve_time_ms=_format_float(duration_ms),
            validation_warnings=[],
            recommendation=recommendation,
            max_positive_moment=_format_float(max_positive[1]) if max_positive else None,
            max_positive_position=_format_float(max_positive[0]) if max_positive else None,
            min_negative_moment=_format_float(min_negative[1]) if min_negative else None,
            min_negative_position=_format_float(min_negative[0]) if min_negative else None,
            max_absolute_moment=_format_float(max_absolute[1]) if max_absolute else None,
            max_absolute_position=_format_float(max_absolute[0]) if max_absolute else None,
        ),
    )


def _compute_cantilever_reactions(payload: SolveRequest) -> List[SupportReaction]:
    """Resolve the single fixed support reactions for a cantilever beam."""
    support = payload.supports[0]
    total_vertical = 0.0
    total_axial = 0.0
    total_moment_about_support = 0.0

    for load in payload.point_loads:
        vertical = _vertical_component(load)
        axial = _axial_component(load)
        total_vertical += vertical
        total_axial += axial
        lever = load.position - support.position
        total_moment_about_support += vertical * lever

    for udl in payload.udls:
        equivalent_force, centroid = _udl_equivalent_force_and_centroid(udl)
        total_vertical += equivalent_force
        total_moment_about_support += equivalent_force * (centroid - support.position)

    for moment in payload.moment_loads:
        total_moment_about_support += moment.magnitude * _moment_sign(moment.direction)

    reaction_vertical = total_vertical
    reaction_axial = -total_axial
    reaction_moment = -total_moment_about_support

    reactions = [
        SupportReaction(
            support_id=support.id,
            support_type=support.type,
            position=_format_float(support.position),
            vertical=_format_float(reaction_vertical),
            axial=_format_float(reaction_axial),
            moment=_format_float(reaction_moment),
        )
    ]

    return reactions


def _cantilever_normal_diagram(payload: SolveRequest, x_axis: np.ndarray, reactions: List[SupportReaction]) -> np.ndarray:
    """Build the axial force diagram for a cantilever."""
    normal = np.zeros_like(x_axis, dtype=float)

    for reaction in reactions:
        normal += reaction.axial * (x_axis >= reaction.position)

    for load in payload.point_loads:
        axial = _axial_component(load)
        normal -= axial * (x_axis >= load.position)

    return normal


def _build_cantilever_axis(payload: SolveRequest, reactions: List[SupportReaction]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate x, shear and moment arrays with refined sampling near jumps."""
    base_axis = np.linspace(0.0, payload.length, num=DEFAULT_SAMPLING_POINTS, dtype=float, endpoint=True)
    if base_axis.size > 0:
        base_axis[0] = 0.0
        base_axis[-1] = payload.length

    shear_base = _shear_diagram(payload, base_axis, reactions)
    critical_points: List[float] = base_axis.tolist()

    for support in payload.supports:
        _add_unique_point(critical_points, support.position, payload.length)

    for load in payload.point_loads:
        _add_unique_point(critical_points, load.position, payload.length)

    for udl in payload.udls:
        _add_unique_point(critical_points, udl.start, payload.length)
        _add_unique_point(critical_points, udl.end, payload.length)
        span = udl.end - udl.start
        if span > 0:
            for fraction in (0.25, 0.5, 0.75):
                _add_unique_point(critical_points, udl.start + fraction * span, payload.length)

    for moment_load in payload.moment_loads:
        _add_unique_point(critical_points, moment_load.position, payload.length)

    for idx in range(len(base_axis) - 1):
        left = base_axis[idx]
        right = base_axis[idx + 1]
        s_left = shear_base[idx]
        s_right = shear_base[idx + 1]

        if abs(s_left) < ROOT_TOL:
            _add_unique_point(critical_points, left, payload.length)
        if abs(s_right) < ROOT_TOL:
            _add_unique_point(critical_points, right, payload.length)

        if s_left * s_right < 0.0:
            root = _locate_shear_zero(payload, reactions, left, right, s_left, s_right)
            _add_unique_point(critical_points, root, payload.length)
        else:
            mid = 0.5 * (left + right)
            s_mid = float(_shear_diagram(payload, np.array([mid], dtype=float), reactions)[0])
            if s_left * s_mid < 0.0:
                root = _locate_shear_zero(payload, reactions, left, mid, s_left, s_mid)
                _add_unique_point(critical_points, root, payload.length)
            elif s_mid * s_right < 0.0:
                root = _locate_shear_zero(payload, reactions, mid, right, s_mid, s_right)
                _add_unique_point(critical_points, root, payload.length)

    x_axis = np.array(sorted(critical_points), dtype=float)

    shear = _shear_diagram(payload, x_axis, reactions)
    normal = _cantilever_normal_diagram(payload, x_axis, reactions)
    moment = _moment_diagram(payload, x_axis, reactions)

    discontinuity_positions: List[float] = []
    for reaction in reactions:
        if abs(reaction.vertical) > ROOT_TOL:
            discontinuity_positions.append(reaction.position)
    for load in payload.point_loads:
        vertical = _vertical_component(load)
        if abs(vertical) > ROOT_TOL:
            discontinuity_positions.append(load.position)

    if discontinuity_positions:
        x_axis_refined: List[float] = []
        shear_refined: List[float] = []
        normal_refined: List[float] = []
        moment_refined: List[float] = []

        for idx, x_val in enumerate(x_axis):
            is_jump = any(math.isclose(x_val, pos, abs_tol=ROOT_TOL, rel_tol=0.0) for pos in discontinuity_positions)
            if is_jump:
                left_eval = float(np.nextafter(x_val, -np.inf))
                shear_left = float(_shear_diagram(payload, np.array([left_eval], dtype=float), reactions)[0])
                normal_left = float(_cantilever_normal_diagram(payload, np.array([left_eval], dtype=float), reactions)[0])
                moment_left = float(_moment_diagram(payload, np.array([left_eval], dtype=float), reactions)[0])
                x_axis_refined.append(float(x_val))
                shear_refined.append(shear_left)
                normal_refined.append(normal_left)
                moment_refined.append(moment_left)

            x_axis_refined.append(float(x_val))
            shear_refined.append(float(shear[idx]))
            normal_refined.append(float(normal[idx]))
            moment_refined.append(float(moment[idx]))

        x_axis = np.array(x_axis_refined, dtype=float)
        shear = np.array(shear_refined, dtype=float)
        normal = np.array(normal_refined, dtype=float)
        moment = np.array(moment_refined, dtype=float)

    return x_axis, shear, moment


def solve_cantilever_beam(payload: SolveRequest) -> SolveResponse:
    """Entry point that solves cantilever reactions and diagrams only."""
    start_time = perf_counter()
    reactions = _compute_cantilever_reactions(payload)
    recommendation = _determine_method_recommendation(payload)

    x_axis, shear, moment = _build_cantilever_axis(payload, reactions)
    moment_extrema = _compute_moment_extrema(payload, reactions, x_axis, shear)

    warnings: List[str] = []
    max_positive = moment_extrema.get("max_positive")
    min_negative = moment_extrema.get("min_negative")
    max_absolute = moment_extrema.get("max_absolute")

    duration_ms = (perf_counter() - start_time) * 1000.0

    diagram_data = DiagramData(
        x=[_format_float(value) for value in x_axis.tolist()],
        shear=[_format_float(value) for value in shear.tolist()],
        moment=[_format_float(value) for value in moment.tolist()],
        normal=[0.0 for _ in x_axis.tolist()],
    )

    return SolveResponse(
        reactions=[
            SupportReaction(
                support_id=reaction.support_id,
                support_type=reaction.support_type,
                position=_format_float(reaction.position),
                vertical=_format_float(reaction.vertical),
                axial=_format_float(reaction.axial),
                moment=_format_float(reaction.moment),
            )
            for reaction in reactions
        ],
        diagram=diagram_data,
        meta=SolveMeta(
            solve_time_ms=_format_float(duration_ms),
            validation_warnings=warnings,
            recommendation=recommendation,
            max_positive_moment=_format_float(max_positive[1]) if max_positive else None,
            max_positive_position=_format_float(max_positive[0]) if max_positive else None,
            min_negative_moment=_format_float(min_negative[1]) if min_negative else None,
            min_negative_position=_format_float(min_negative[0]) if min_negative else None,
            max_absolute_moment=_format_float(max_absolute[1]) if max_absolute else None,
            max_absolute_position=_format_float(max_absolute[0]) if max_absolute else None,
        ),
    )


def calculate_fundamental_period(payload: ChimneyPeriodRequest) -> ChimneyPeriodResponse:
    """Compute the first mode period/frequency for a cantilever-like chimney."""
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
