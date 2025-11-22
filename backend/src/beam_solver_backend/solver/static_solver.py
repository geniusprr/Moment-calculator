from __future__ import annotations

import math
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from beam_solver_backend.schemas.beam import (
    DiagramData,
    MethodRecommendation,
    SolveMeta,
    SolveRequest,
    SolveResponse,
    SupportReaction,
)
from beam_solver_backend.solver.detailed_solver import DetailedSolver

DEFAULT_SAMPLING_POINTS = 401
ROOT_TOL = 1e-9

MomentCandidate = Tuple[float, float]

def _format_float(value: float) -> float:
    return float(f"{value:.6f}")

def _vertical_component(load) -> float:
    angle_rad = math.radians(load.angle_deg)
    vertical = -load.magnitude * math.sin(angle_rad)
    if abs(vertical) < 1e-9:
        return 0.0
    return vertical

def _axial_component(load) -> float:
    angle_rad = math.radians(load.angle_deg)
    axial = load.magnitude * math.cos(angle_rad)
    if abs(axial) < 1e-9:
        return 0.0
    return axial

def _udl_sign(udl) -> float:
    return 1.0 if udl.direction == "down" else -1.0

def _udl_equivalent_force_and_centroid(udl) -> tuple[float, float]:
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
    if math.isnan(value) or math.isinf(value):
        return
    clamped = min(max(value, 0.0), beam_length)
    for existing in points:
        if math.isclose(existing, clamped, abs_tol=tol, rel_tol=0.0):
            return
    points.append(clamped)

def _moment_value(payload: SolveRequest, reactions: List[SupportReaction], position: float) -> float:
    clamped = min(max(position, 0.0), payload.length)
    return float(_moment_diagram(payload, np.array([clamped], dtype=float), reactions)[0])

def _register_moment_candidate(
    candidates: List[MomentCandidate],
    payload: SolveRequest,
    reactions: List[SupportReaction],
    position: float,
    tol: float = 1e-6,
) -> None:
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
    return MethodRecommendation(
        method="area",
        title="Alan Yontemi",
        reason="Standart cozum yontemi.",
    )

def _moment_sign(direction) -> float:
    return 1.0 if direction == "ccw" else -1.0

def _compute_reactions(payload: SolveRequest) -> tuple[List[SupportReaction], List[str]]:
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

    derivations = [
        f"Toplam Yuk: {total_vertical:.2f} kN",
        f"A Mesnetine Gore Moment: {total_moment_about_a:.2f} kN.m",
    ]

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

    return reactions, derivations

def _shear_diagram(payload: SolveRequest, x_axis: np.ndarray, reactions: List[SupportReaction]) -> np.ndarray:
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
    normal = np.zeros_like(x_axis, dtype=float)

    for reaction in reactions:
        normal += reaction.axial * (x_axis >= reaction.position)

    for load in payload.point_loads:
        axial = _axial_component(load)
        normal += axial * (x_axis >= load.position)

    return normal

def _moment_diagram(payload: SolveRequest, x_axis: np.ndarray, reactions: List[SupportReaction]) -> np.ndarray:
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
    start_time = perf_counter()
    reactions, derivations = _compute_reactions(payload)
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

    # Generate detailed solution
    diagram_data = DiagramData(
        x=[_format_float(v) for v in x_axis],
        shear=[_format_float(v) for v in shear],
        moment=[_format_float(v) for v in moment],
        normal=[_format_float(v) for v in normal],
    )
    
    detailed_solver = DetailedSolver(payload, reactions, diagram_data)
    detailed_solution = detailed_solver.solve()

    return SolveResponse(
        reactions=reactions,
        diagram=diagram_data,
        derivations=derivations,
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
        detailed_solutions=detailed_solution
    )

