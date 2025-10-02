from __future__ import annotations

import math
from time import perf_counter
from typing import List

import numpy as np

from beam_solver_backend.schemas.beam import (
    DiagramData,
    MomentDirection,
    SolveRequest,
    SolveResponse,
    SolveMeta,
    SupportReaction,
)


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


def _udl_intensity(udl) -> float:
    return udl.magnitude if udl.direction == "down" else -udl.magnitude


def _moment_sign(direction: MomentDirection) -> float:
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
        intensity = _udl_intensity(udl)
        span_length = udl.end - udl.start
        equivalent_force = intensity * span_length
        centroid = udl.start + span_length / 2.0
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
        rf"\\sum F_y = 0: R_{{Ay}} + R_{{By}} = {total_vertical:.3f}",
        rf"\\sum M_A = 0: R_{{By}} \times {span:.3f} = {total_moment_about_a:.3f}",
        rf"R_{{Ay}} = {reaction_a_vertical:.3f},\\quad R_{{By}} = {reaction_b_vertical:.3f}",
    ]

    if abs(total_axial) > 1e-9:
        derivations.append(rf"\\sum F_x = 0: R_{{Ax}} = {-total_axial:.3f}")

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
        intensity = _udl_intensity(udl)
        span = udl.end - udl.start
        contribution = np.clip(x_axis - udl.start, 0.0, span)
        shear -= intensity * contribution

    return shear


def _normal_diagram(payload: SolveRequest, x_axis: np.ndarray, reactions: List[SupportReaction]) -> np.ndarray:
    normal = np.zeros_like(x_axis, dtype=float)

    for reaction in reactions:
        normal += reaction.axial * (x_axis >= reaction.position)

    for load in payload.point_loads:
        axial = _axial_component(load)
        normal += axial * (x_axis >= load.position)

    return normal


def _moment_diagram(x_axis: np.ndarray, shear: np.ndarray) -> np.ndarray:
    moment = np.zeros_like(shear)
    increments = np.diff(x_axis)
    trapezoids = 0.5 * (shear[:-1] + shear[1:]) * increments
    moment[1:] = np.cumsum(trapezoids)
    return moment


def _apply_concentrated_moments(moment: np.ndarray, x_axis: np.ndarray, payload: SolveRequest) -> np.ndarray:
    adjusted = moment.copy()
    for load in payload.moment_loads:
        signed = load.magnitude * _moment_sign(load.direction)
        adjusted += signed * (x_axis >= load.position)
    return adjusted


def solve_beam(payload: SolveRequest) -> SolveResponse:
    sampling_points = payload.sampling.points if payload.sampling else 401

    start_time = perf_counter()
    reactions, derivations = _compute_reactions(payload)

    x_axis = np.linspace(0.0, payload.length, num=sampling_points, dtype=float)
    shear = _shear_diagram(payload, x_axis, reactions)
    normal = _normal_diagram(payload, x_axis, reactions)
    moment = _moment_diagram(x_axis, shear)
    moment = _apply_concentrated_moments(moment, x_axis, payload)

    warnings: List[str] = []

    supports_sorted = sorted(payload.supports, key=lambda support: support.position)
    right_support_pos = supports_sorted[-1].position
    moment_at_right = float(np.interp(right_support_pos, x_axis, moment))
    if abs(moment_at_right) > 1e-2:
        warnings.append(
            f"Moment at x={right_support_pos:.2f} m is not close to zero (|{moment_at_right:.3f}|). Numerical drift may be present."
        )

    axial_balance = sum(reaction.axial for reaction in reactions) - sum(
        _axial_component(load) for load in payload.point_loads
    )
    if abs(axial_balance) > 1e-3:
        warnings.append("Axial equilibrium residual is larger than expected.")

    duration_ms = (perf_counter() - start_time) * 1000.0

    response = SolveResponse(
        reactions=[
            SupportReaction(
                support_id=reaction.support_id,
                support_type=reaction.support_type,
                position=_format_float(reaction.position),
                vertical=_format_float(reaction.vertical),
                axial=_format_float(reaction.axial),
            )
            for reaction in reactions
        ],
        diagram=DiagramData(
            x=[_format_float(value) for value in x_axis.tolist()],
            shear=[_format_float(value) for value in shear.tolist()],
            moment=[_format_float(value) for value in moment.tolist()],
            normal=[_format_float(value) for value in normal.tolist()],
        ),
        derivations=derivations,
        meta=SolveMeta(
            solve_time_ms=_format_float(duration_ms),
            validation_warnings=warnings,
        ),
    )

    return response

