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
from beam_solver_backend.solver import static_solver as base
from beam_solver_backend.solver.detailed_solver import DetailedSolver

# Reuse helper constants/functions from the existing solver for consistency
MomentCandidate = Tuple[float, float]
DEFAULT_SAMPLING_POINTS = base.DEFAULT_SAMPLING_POINTS
ROOT_TOL = base.ROOT_TOL
_format_float = base._format_float
_vertical_component = base._vertical_component
_axial_component = base._axial_component
_udl_equivalent_force_and_centroid = base._udl_equivalent_force_and_centroid
_udl_shear_contribution = base._udl_shear_contribution
_udl_moment_contribution = base._udl_moment_contribution
_moment_sign = base._moment_sign

def _compute_cantilever_reactions(payload: SolveRequest) -> tuple[List[SupportReaction], List[str]]:
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

    derivations = [
        f"Toplam Dusey Yuk: {total_vertical:.2f} kN",
        f"Ankastre Moment: {reaction_moment:.2f} kN.m",
    ]

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
        normal -= axial * (x_axis >= load.position)

    return normal

def _moment_diagram(payload: SolveRequest, x_axis: np.ndarray, reactions: List[SupportReaction]) -> np.ndarray:
    return base._moment_diagram(payload, x_axis, reactions)

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
    return base._compute_moment_extrema(payload, reactions, x_axis, shear)

def _build_axis(payload: SolveRequest, reactions: List[SupportReaction]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sampling_points = DEFAULT_SAMPLING_POINTS
    base_axis = np.linspace(0.0, payload.length, num=sampling_points, dtype=float, endpoint=True)
    if base_axis.size > 0:
        base_axis[0] = 0.0
        base_axis[-1] = payload.length

    shear_base = _shear_diagram(payload, base_axis, reactions)
    critical_points: List[float] = base_axis.tolist()

    for support in payload.supports:
        base._add_unique_point(critical_points, support.position, payload.length)

    for load in payload.point_loads:
        base._add_unique_point(critical_points, load.position, payload.length)

    for udl in payload.udls:
        base._add_unique_point(critical_points, udl.start, payload.length)
        base._add_unique_point(critical_points, udl.end, payload.length)
        span = udl.end - udl.start
        if span > 0:
            for fraction in (0.25, 0.5, 0.75):
                base._add_unique_point(critical_points, udl.start + fraction * span, payload.length)

    for moment_load in payload.moment_loads:
        base._add_unique_point(critical_points, moment_load.position, payload.length)

    for idx in range(len(base_axis) - 1):
        left = base_axis[idx]
        right = base_axis[idx + 1]
        s_left = shear_base[idx]
        s_right = shear_base[idx + 1]

        if abs(s_left) < ROOT_TOL:
            base._add_unique_point(critical_points, left, payload.length)
        if abs(s_right) < ROOT_TOL:
            base._add_unique_point(critical_points, right, payload.length)

        if s_left * s_right < 0.0:
            root = base._locate_shear_zero(payload, reactions, left, right, s_left, s_right)
            base._add_unique_point(critical_points, root, payload.length)
        else:
            mid = 0.5 * (left + right)
            s_mid = float(_shear_diagram(payload, np.array([mid], dtype=float), reactions)[0])
            if s_left * s_mid < 0.0:
                root = base._locate_shear_zero(payload, reactions, left, mid, s_left, s_mid)
                base._add_unique_point(critical_points, root, payload.length)
            elif s_mid * s_right < 0.0:
                root = base._locate_shear_zero(payload, reactions, mid, right, s_mid, s_right)
                base._add_unique_point(critical_points, root, payload.length)

    x_axis = np.array(sorted(critical_points), dtype=float)

    shear = _shear_diagram(payload, x_axis, reactions)
    normal = _normal_diagram(payload, x_axis, reactions)
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
                normal_left = float(_normal_diagram(payload, np.array([left_eval], dtype=float), reactions)[0])
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
    start_time = perf_counter()
    reactions, derivations = _compute_cantilever_reactions(payload)
    recommendation = base._determine_method_recommendation(payload)

    x_axis, shear, moment = _build_axis(payload, reactions)
    moment_extrema = _compute_moment_extrema(payload, reactions, x_axis, shear)

    warnings: List[str] = []
    max_positive = moment_extrema.get("max_positive")
    min_negative = moment_extrema.get("min_negative")
    max_absolute = moment_extrema.get("max_absolute")

    duration_ms = (perf_counter() - start_time) * 1000.0

    # Generate detailed solution
    diagram_data = DiagramData(
        x=[_format_float(value) for value in x_axis.tolist()],
        shear=[_format_float(value) for value in shear.tolist()],
        moment=[_format_float(value) for value in moment.tolist()],
        normal=[0.0 for _ in x_axis.tolist()],
    )
    
    detailed_solver = DetailedSolver(payload, reactions, diagram_data)
    detailed_solution = detailed_solver.solve()

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
        derivations=derivations,
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
        detailed_solutions=detailed_solution,
    )

