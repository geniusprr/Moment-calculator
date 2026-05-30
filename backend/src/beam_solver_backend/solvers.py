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
    DetailedSolution,
    SolutionMethod,
    SolutionStep,
    BeamSectionHighlight,
    AreaMethodVisualization,
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


def _moment_diagram(payload: SolveRequest, x_axis: np.ndarray, reactions: List[SupportReaction]) -> np.ndarray:
    """Integrate shear effects and applied moments to obtain bending moment values."""
    moment = np.zeros_like(x_axis, dtype=float)

    for reaction in reactions:
        offsets = np.maximum(x_axis - reaction.position, 0.0)
        moment += reaction.vertical * offsets
        if getattr(reaction, "moment", 0.0) != 0.0:
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


def macaulay(x: np.ndarray, a: float, n: int) -> np.ndarray:
    """Evaluate Macaulay bracket discontinuity function <x - a>^n."""
    mask = x >= a
    if n == 0:
        return np.where(mask, 1.0, 0.0)
    return np.where(mask, (x - a) ** n, 0.0)


def get_applied_loads_at_x(x: float, payload: SolveRequest) -> Tuple[float, float, float]:
    """Evaluate applied load bending moment, EI*theta and EI*w at coordinate x."""
    x_arr = np.array([x], dtype=float)
    moment = 0.0
    theta = 0.0
    w = 0.0

    # Point loads
    for load in payload.point_loads:
        p_val = _vertical_component(load)  # downward positive
        pos = load.position
        moment -= p_val * macaulay(x_arr, pos, 1)[0]
        theta += (p_val / 2.0) * macaulay(x_arr, pos, 2)[0]
        w += (p_val / 6.0) * macaulay(x_arr, pos, 3)[0]

    # Moment loads
    for m_load in payload.moment_loads:
        signed_t = m_load.magnitude * _moment_sign(m_load.direction)  # CCW positive
        pos = m_load.position
        moment += signed_t * macaulay(x_arr, pos, 0)[0]
        theta -= signed_t * macaulay(x_arr, pos, 1)[0]
        w -= (signed_t / 2.0) * macaulay(x_arr, pos, 2)[0]

    # UDLs
    for udl in payload.udls:
        q = udl.magnitude * _udl_sign(udl)  # downward positive
        a_pos, b_pos = udl.start, udl.end
        L_u = b_pos - a_pos
        if L_u <= 0:
            continue
        if udl.shape == "uniform":
            moment -= (q / 2.0) * (macaulay(x_arr, a_pos, 2)[0] - macaulay(x_arr, b_pos, 2)[0])
            theta += (q / 6.0) * (macaulay(x_arr, a_pos, 3)[0] - macaulay(x_arr, b_pos, 3)[0])
            w += (q / 24.0) * (macaulay(x_arr, a_pos, 4)[0] - macaulay(x_arr, b_pos, 4)[0])
        elif udl.shape == "triangular_increasing":
            moment -= (q / (6.0 * L_u)) * (macaulay(x_arr, a_pos, 3)[0] - macaulay(x_arr, b_pos, 3)[0]) + (q / 2.0) * macaulay(x_arr, b_pos, 2)[0]
            theta += (q / (24.0 * L_u)) * (macaulay(x_arr, a_pos, 4)[0] - macaulay(x_arr, b_pos, 4)[0]) - (q / 6.0) * macaulay(x_arr, b_pos, 3)[0]
            w += (q / (120.0 * L_u)) * (macaulay(x_arr, a_pos, 5)[0] - macaulay(x_arr, b_pos, 5)[0]) - (q / 24.0) * macaulay(x_arr, b_pos, 4)[0]
        elif udl.shape == "triangular_decreasing":
            moment_udl = (q / 2.0) * (macaulay(x_arr, a_pos, 2)[0] - macaulay(x_arr, b_pos, 2)[0])
            theta_udl = (q / 6.0) * (macaulay(x_arr, a_pos, 3)[0] - macaulay(x_arr, b_pos, 3)[0])
            w_udl = (q / 24.0) * (macaulay(x_arr, a_pos, 4)[0] - macaulay(x_arr, b_pos, 4)[0])

            moment_inc = (q / (6.0 * L_u)) * (macaulay(x_arr, a_pos, 3)[0] - macaulay(x_arr, b_pos, 3)[0]) + (q / 2.0) * macaulay(x_arr, b_pos, 2)[0]
            theta_inc = (q / (24.0 * L_u)) * (macaulay(x_arr, a_pos, 4)[0] - macaulay(x_arr, b_pos, 4)[0]) - (q / 6.0) * macaulay(x_arr, b_pos, 3)[0]
            w_inc = (q / (120.0 * L_u)) * (macaulay(x_arr, a_pos, 5)[0] - macaulay(x_arr, b_pos, 5)[0]) - (q / 24.0) * macaulay(x_arr, b_pos, 4)[0]

            moment -= (moment_udl - moment_inc)
            theta += (theta_udl - theta_inc)
            w += (w_udl - w_inc)

    return moment, theta, w


def build_refined_axis(payload: SolveRequest, supports_positions: List[float]) -> np.ndarray:
    """Generate x values with extra samples around supports/loads."""
    base_axis = np.linspace(0.0, payload.length, num=DEFAULT_SAMPLING_POINTS, dtype=float, endpoint=True)
    critical_points = base_axis.tolist()
    for pos in supports_positions:
        _add_unique_point(critical_points, pos, payload.length)
    for load in payload.point_loads:
        _add_unique_point(critical_points, load.position, payload.length)
    for udl in payload.udls:
        _add_unique_point(critical_points, udl.start, payload.length)
        _add_unique_point(critical_points, udl.end, payload.length)
        span = udl.end - udl.start
        if span > 0:
            for fraction in (0.25, 0.5, 0.75):
                _add_unique_point(critical_points, udl.start + fraction * span, payload.length)
    for m in payload.moment_loads:
        _add_unique_point(critical_points, m.position, payload.length)
    return np.array(sorted(list(set(critical_points))), dtype=float)


def evaluate_diagrams(
    x_axis: np.ndarray,
    payload: SolveRequest,
    reactions: List[SupportReaction],
    C1: float,
    C2: float,
    EI: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute V(x), M(x), N(x), theta(x), and w(x) arrays."""
    shear = np.zeros_like(x_axis, dtype=float)
    moment = np.zeros_like(x_axis, dtype=float)
    normal = np.zeros_like(x_axis, dtype=float)
    theta_ei = np.zeros_like(x_axis, dtype=float)
    w_ei = np.zeros_like(x_axis, dtype=float)

    # 1. Reactions
    for r in reactions:
        pos = r.position
        shear += r.vertical * macaulay(x_axis, pos, 0)
        moment += r.vertical * macaulay(x_axis, pos, 1)
        normal += r.axial * macaulay(x_axis, pos, 0)
        theta_ei -= (r.vertical / 2.0) * macaulay(x_axis, pos, 2)
        w_ei -= (r.vertical / 6.0) * macaulay(x_axis, pos, 3)

        if r.support_type == "fixed":
            moment += r.moment * macaulay(x_axis, pos, 0)
            theta_ei -= r.moment * macaulay(x_axis, pos, 1)
            w_ei -= (r.moment / 2.0) * macaulay(x_axis, pos, 2)

    # 2. Applied Point Loads
    for load in payload.point_loads:
        p_val = _vertical_component(load)
        axial = _axial_component(load)
        pos = load.position
        shear -= p_val * macaulay(x_axis, pos, 0)
        moment -= p_val * macaulay(x_axis, pos, 1)
        normal += axial * macaulay(x_axis, pos, 0)
        theta_ei += (p_val / 2.0) * macaulay(x_axis, pos, 2)
        w_ei += (p_val / 6.0) * macaulay(x_axis, pos, 3)

    # 3. Applied Moments
    for m in payload.moment_loads:
        signed_t = m.magnitude * _moment_sign(m.direction)
        pos = m.position
        moment += signed_t * macaulay(x_axis, pos, 0)
        theta_ei -= signed_t * macaulay(x_axis, pos, 1)
        w_ei -= (signed_t / 2.0) * macaulay(x_axis, pos, 2)

    # 4. Applied UDLs
    for udl in payload.udls:
        q = udl.magnitude * _udl_sign(udl)
        a, b = udl.start, udl.end
        L_u = b - a
        if L_u <= 0:
            continue
        if udl.shape == "uniform":
            shear -= q * (macaulay(x_axis, a, 1) - macaulay(x_axis, b, 1))
            moment -= (q / 2.0) * (macaulay(x_axis, a, 2) - macaulay(x_axis, b, 2))
            theta_ei += (q / 6.0) * (macaulay(x_axis, a, 3) - macaulay(x_axis, b, 3))
            w_ei += (q / 24.0) * (macaulay(x_axis, a, 4) - macaulay(x_axis, b, 4))
        elif udl.shape == "triangular_increasing":
            shear -= (q / (2.0 * L_u)) * (macaulay(x_axis, a, 2) - macaulay(x_axis, b, 2)) + q * macaulay(x_axis, b, 1)
            moment -= (q / (6.0 * L_u)) * (macaulay(x_axis, a, 3) - macaulay(x_axis, b, 3)) + (q / 2.0) * macaulay(x_axis, b, 2)
            theta_ei += (q / (24.0 * L_u)) * (macaulay(x_axis, a, 4) - macaulay(x_axis, b, 4)) - (q / 6.0) * macaulay(x_axis, b, 3)
            w_ei += (q / (120.0 * L_u)) * (macaulay(x_axis, a, 5) - macaulay(x_axis, b, 5)) - (q / 24.0) * macaulay(x_axis, b, 4)
        elif udl.shape == "triangular_decreasing":
            shear_udl = q * (macaulay(x_axis, a, 1) - macaulay(x_axis, b, 1))
            shear_inc = (q / (2.0 * L_u)) * (macaulay(x_axis, a, 2) - macaulay(x_axis, b, 2)) + q * macaulay(x_axis, b, 1)
            shear -= (shear_udl - shear_inc)

            moment_udl = (q / 2.0) * (macaulay(x_axis, a, 2) - macaulay(x_axis, b, 2))
            moment_inc = (q / (6.0 * L_u)) * (macaulay(x_axis, a, 3) - macaulay(x_axis, b, 3)) + (q / 2.0) * macaulay(x_axis, b, 2)
            moment -= (moment_udl - moment_inc)

            theta_udl = (q / 6.0) * (macaulay(x_axis, a, 3) - macaulay(x_axis, b, 3))
            theta_inc = (q / (24.0 * L_u)) * (macaulay(x_axis, a, 4) - macaulay(x_axis, b, 4)) - (q / 6.0) * macaulay(x_axis, b, 3)
            theta_ei += (theta_udl - theta_inc)

            w_udl = (q / 24.0) * (macaulay(x_axis, a, 4) - macaulay(x_axis, b, 4))
            w_inc = (q / (120.0 * L_u)) * (macaulay(x_axis, a, 5) - macaulay(x_axis, b, 5)) - (q / 24.0) * macaulay(x_axis, b, 4)
            w_ei += (w_udl - w_inc)

    theta = (theta_ei + C1) / EI
    w = (w_ei + C1 * x_axis + C2) / EI
    deflection_mm = w * 1000.0

    return shear, moment, normal, theta, deflection_mm


def generate_detailed_solution_steps(
    payload: SolveRequest,
    reactions: List[SupportReaction],
    C1: float,
    C2: float,
    E: float,
    I: float,
    max_defl: float,
    max_defl_pos: float,
) -> DetailedSolution:
    """Generate step by step explanations with LaTeX formulas in Turkish."""
    reaction_steps = []

    # Step 1: Serbest Cisim Diyagramı
    explanation_1 = (
        "Statik ve mukavemet analizinin ilk adımı, kiriş üzerindeki tüm aktif yükleri ve sınır koşullarını "
        "içeren Serbest Cisim Diyagramını (SCD) tanımlamaktır. Bu aşamada dikey kuvvet dengesi (\\sum F_y = 0) "
        "ve moment dengesi (\\sum M = 0) şartları tanımlanır."
    )
    reaction_steps.append(
        SolutionStep(
            step_number=1,
            title="1. Serbest Cisim Diyagramı ve Denge Şartları",
            explanation=explanation_1,
            general_formula=r"\sum F_y = 0, \quad \sum M = 0",
        )
    )

    # Step 2: Denge Denklemleri
    explanation_2 = (
        "Kirişe etkiyen dikey yüklerin ve reaksiyon kuvvetlerinin toplamını sıfıra eşitliyoruz. "
        "Ayrıca dikey mesnet reaksiyon kuvvetleri dikey statik denge denklemleriyle formüle edilir."
    )
    vert_lhs = " + ".join([f"R_{{{r.support_id}}}" for r in reactions])
    total_p = sum(_vertical_component(load) for load in payload.point_loads)
    total_udl = 0.0
    for udl in payload.udls:
        span = udl.end - udl.start
        q = udl.magnitude * _udl_sign(udl)
        if udl.shape == "uniform":
            total_udl += q * span
        else:
            total_udl += 0.5 * q * span
    formula_2 = f"{vert_lhs} - ({total_p + total_udl:.2f}) = 0"
    reaction_steps.append(
        SolutionStep(
            step_number=2,
            title="2. Denge Denkleminin Kurulması",
            explanation=explanation_2,
            general_formula=r"\sum R_i - \sum P_{y,k} - \sum W_{udl,m} = 0",
            substituted_formula=formula_2,
        )
    )

    # Step 3: Çözüm
    explanation_3 = (
        "Macaulay sınır koşulları ile statik denge şartları birleştirilerek dikey mesnet tepki kuvvetleri "
        "ve sabitleme momentleri kesin olarak elde edilir."
    )
    res_lines = []
    for r in reactions:
        res_lines.append(f"R_{{{r.support_id}}} = {r.vertical:.2f} \\text{{ kN}} \\quad (x = {r.position:.2f}\\text{{ m}})")
        if r.support_type == "fixed":
            res_lines.append(f"M_{{{r.support_id}}} = {r.moment:.2f} \\text{{ kNm}}")
    reaction_steps.append(
        SolutionStep(
            step_number=3,
            title="3. Reaksiyon Değerlerinin Hesaplanması",
            explanation=explanation_3,
            numerical_result=" \\\\ ".join(res_lines),
        )
    )

    integration_steps = []

    # Step 1: Moment Denklemi
    explanation_int_1 = (
        "Macaulay Yöntemi ile kiriş boyunca tek bir sürekli moment denklemi M(x) yazılır. "
        "Macaulay parantezleri \\langle x - a \\rangle^n, x < a için sıfırdır, x \\ge a için normal parantez görevi görür."
    )
    m_terms = []
    for r in reactions:
        m_terms.append(f"{r.vertical:+.2f} \\langle x - {r.position:.2f} \\rangle^1")
        if r.support_type == "fixed":
            m_terms.append(f"{r.moment:+.2f} \\langle x - {r.position:.2f} \\rangle^0")
    for load in payload.point_loads:
        p_val = _vertical_component(load)
        m_terms.append(f"{-p_val:+.2f} \\langle x - {load.position:.2f} \\rangle^1")
    for ml in payload.moment_loads:
        sig_t = ml.magnitude * _moment_sign(ml.direction)
        m_terms.append(f"{sig_t:+.2f} \\langle x - {ml.position:.2f} \\rangle^0")
    for udl in payload.udls:
        q = udl.magnitude * _udl_sign(udl)
        a, b = udl.start, udl.end
        Lu = b - a
        if udl.shape == "uniform":
            m_terms.append(f"{-q/2.0:+.2f} \\langle x - {a:.2f} \\rangle^2 {+q/2.0:+.2f} \\langle x - {b:.2f} \\rangle^2")
        elif udl.shape == "triangular_increasing":
            m_terms.append(f"{-q/(6.0*Lu):+.4f} \\langle x - {a:.2f} \\rangle^3 {+q/(6.0*Lu):+.4f} \\langle x - {b:.2f} \\rangle^3 {-q/2.0:+.2f} \\langle x - {b:.2f} \\rangle^2")
        elif udl.shape == "triangular_decreasing":
            m_terms.append(f"{-q/2.0:+.2f} \\langle x - {a:.2f} \\rangle^2 {+q/2.0:+.2f} \\langle x - {b:.2f} \\rangle^2")
            m_terms.append(f"{+q/(6.0*Lu):+.4f} \\langle x - {a:.2f} \\rangle^3 {-q/(6.0*Lu):+.4f} \\langle x - {b:.2f} \\rangle^3 {+q/2.0:+.2f} \\langle x - {b:.2f} \\rangle^2")
    m_eq = " ".join(m_terms)

    integration_steps.append(
        SolutionStep(
            step_number=1,
            title="1. Macaulay Eğilme Momenti Fonksiyonu M(x)",
            explanation=explanation_int_1,
            general_formula=r"M(x) = \sum R_i \langle x - s_i \rangle^1 + \sum M_{R,j} \langle x - f_j \rangle^0 - \sum P_k \langle x - p_k \rangle^1 \dots",
            substituted_formula=f"M(x) = {m_eq}",
        )
    )

    # Step 2: Eğim Denklemi
    explanation_int_2 = (
        "Moment denkleminin integrali alınarak eğim (dönme) fonksiyonu EI*\\theta(x) elde edilir. "
        "Burada C_1 entegrasyon sabitidir. Entegrasyon kuralı: \\int \\langle x - a \\rangle^n dx = \\frac{\\langle x - a \\rangle^{n+1}}{n+1}"
    )
    t_terms = []
    for r in reactions:
        t_terms.append(f"{-r.vertical/2.0:+.2f} \\langle x - {r.position:.2f} \\rangle^2")
        if r.support_type == "fixed":
            t_terms.append(f"{-r.moment:+.2f} \\langle x - {r.position:.2f} \\rangle^1")
    for load in payload.point_loads:
        p_val = _vertical_component(load)
        t_terms.append(f"{p_val/2.0:+.2f} \\langle x - {load.position:.2f} \\rangle^2")
    for ml in payload.moment_loads:
        sig_t = ml.magnitude * _moment_sign(ml.direction)
        t_terms.append(f"{-sig_t:+.2f} \\langle x - {ml.position:.2f} \\rangle^1")
    for udl in payload.udls:
        q = udl.magnitude * _udl_sign(udl)
        a, b = udl.start, udl.end
        Lu = b - a
        if udl.shape == "uniform":
            t_terms.append(f"{q/6.0:+.2f} \\langle x - {a:.2f} \\rangle^3 {-q/6.0:+.2f} \\langle x - {b:.2f} \\rangle^3")
        elif udl.shape == "triangular_increasing":
            t_terms.append(f"{q/(24.0*Lu):+.4f} \\langle x - {a:.2f} \\rangle^4 {-q/(24.0*Lu):+.4f} \\langle x - {b:.2f} \\rangle^4 {-q/6.0:+.2f} \\langle x - {b:.2f} \\rangle^3")
        elif udl.shape == "triangular_decreasing":
            t_terms.append(f"{q/6.0:+.2f} \\langle x - {a:.2f} \\rangle^3 {-q/6.0:+.2f} \\langle x - {b:.2f} \\rangle^3")
            t_terms.append(f"{-q/(24.0*Lu):+.4f} \\langle x - {a:.2f} \\rangle^4 {q/(24.0*Lu):+.4f} \\langle x - {b:.2f} \\rangle^4 {q/6.0:+.2f} \\langle x - {b:.2f} \\rangle^3")
    t_eq = " ".join(t_terms) + " + C_1"

    integration_steps.append(
        SolutionStep(
            step_number=2,
            title="2. Eğim (Dönme) Denklemi EI*\\theta(x)",
            explanation=explanation_int_2,
            general_formula=r"EI \theta(x) = -\int M(x) dx + C_1",
            substituted_formula=f"EI \\theta(x) = {t_eq}",
        )
    )

    # Step 3: Sehim Denklemi
    explanation_int_3 = (
        "Eğim denkleminin tekrar x koordinatına göre integrali alınarak sehim (çökme) fonksiyonu EI*w(x) elde edilir. "
        "C_2 ikinci integrasyon sabitidir. Aşağı yönlü çökme pozitif kabul edilmiştir."
    )
    w_terms = []
    for r in reactions:
        w_terms.append(f"{-r.vertical/6.0:+.2f} \\langle x - {r.position:.2f} \\rangle^3")
        if r.support_type == "fixed":
            w_terms.append(f"{-r.moment/2.0:+.2f} \\langle x - {r.position:.2f} \\rangle^2")
    for load in payload.point_loads:
        p_val = _vertical_component(load)
        w_terms.append(f"{p_val/6.0:+.2f} \\langle x - {load.position:.2f} \\rangle^3")
    for ml in payload.moment_loads:
        sig_t = ml.magnitude * _moment_sign(ml.direction)
        w_terms.append(f"{-sig_t/2.0:+.2f} \\langle x - {ml.position:.2f} \\rangle^2")
    for udl in payload.udls:
        q = udl.magnitude * _udl_sign(udl)
        a, b = udl.start, udl.end
        Lu = b - a
        if udl.shape == "uniform":
            w_terms.append(f"{q/24.0:+.2f} \\langle x - {a:.2f} \\rangle^4 {-q/24.0:+.2f} \\langle x - {b:.2f} \\rangle^4")
        elif udl.shape == "triangular_increasing":
            w_terms.append(f"{q/(120.0*Lu):+.4f} \\langle x - {a:.2f} \\rangle^5 {-q/(120.0*Lu):+.4f} \\langle x - {b:.2f} \\rangle^5 {-q/24.0:+.2f} \\langle x - {b:.2f} \\rangle^4")
        elif udl.shape == "triangular_decreasing":
            w_terms.append(f"{q/24.0:+.2f} \\langle x - {a:.2f} \\rangle^4 {-q/24.0:+.2f} \\langle x - {b:.2f} \\rangle^4")
            w_terms.append(f"{-q/(120.0*Lu):+.4f} \\langle x - {a:.2f} \\rangle^5 {q/(120.0*Lu):+.4f} \\langle x - {b:.2f} \\rangle^5 {q/24.0:+.2f} \\langle x - {b:.2f} \\rangle^4")
    w_eq = " ".join(w_terms) + " + C_1 x + C_2"

    integration_steps.append(
        SolutionStep(
            step_number=3,
            title="3. Sehim (Çökme) Denklemi EI*w(x)",
            explanation=explanation_int_3,
            general_formula=r"EI w(x) = \iint -M(x) dx^2 + C_1 x + C_2",
            substituted_formula=f"EI w(x) = {w_eq}",
        )
    )

    # Step 4: Sınır Koşulları
    explanation_int_4 = (
        "Mesnetlerin olduğu konumlarda dikey çökme sıfırdır (w(s_i) = 0). Ankastre mesnetin olduğu "
        "konumda ise hem dönme hem çökme sıfırdır (\\theta(f_j) = 0, w(f_j) = 0). "
        "Bu sınır şartları uygulanarak bilinmeyen C_1 ve C_2 entegrasyon sabitleri çözülür."
    )
    integration_steps.append(
        SolutionStep(
            step_number=4,
            title="4. Sınır Koşulları ve Entegrasyon Sabitlerinin Çözümü",
            explanation=explanation_int_4,
            general_formula=r"w(s_i) = 0 \implies C_1, C_2 \\ \theta(f_j) = 0 \implies C_1, C_2",
            numerical_result=f"C_1 = {C1:.2f} \\text{{ kN}}\\cdot\\text{{m}}^2 \\quad C_2 = {C2:.2f} \\text{{ kN}}\\cdot\\text{{m}}^3",
        )
    )

    # Step 5: Sehim Sonuç
    explanation_int_5 = (
        f"Kirişin elastik rijitlik değerleri E = {E:.2f} GPa ve I = {I*1e8:.2f} cm^4 "
        f"olduğundan eğilme rijitliği EI = {E*1e6*I:.2f} kN.m^2 olmaktadır. "
        f"Bu değer kullanılarak kirişin maksimum çökme miktarı ve bu çökmenin oluştuğu yer hesaplanmıştır."
    )
    integration_steps.append(
        SolutionStep(
            step_number=5,
            title="5. Maksimum Çökme (Sehim) Sonucu",
            explanation=explanation_int_5,
            general_formula=r"w_{max} = \max|w(x)|",
            numerical_result=f"w_{{max}} = {max_defl:.3f} \\text{{ mm}} \\quad (x = {max_defl_pos:.2f} \\text{{ m}})",
        )
    )

    return DetailedSolution(
        methods=[
            SolutionMethod(
                method_name="support_reactions",
                method_title="1. Mesnet Tepkileri",
                description="Kirişin dengesini sağlayan reaksiyon dikey kuvvetleri ve momentlerinin hesabı.",
                recommended=True,
                steps=reaction_steps,
            ),
            SolutionMethod(
                method_name="integration_method",
                method_title="2. İntegrasyon Yöntemi",
                description="Macaulay tekillik fonksiyonları ile çökme ve dönme eğrisinin analitik hesabı.",
                recommended=True,
                recommendation_reason="Kiriş boyunca çökme (sehim) ve dönme değerlerini tam formülleştirerek verir.",
                steps=integration_steps,
            ),
        ]
    )


def solve_beam_unified(payload: SolveRequest) -> SolveResponse:
    """Unified solver using Macaulay singularity functions to solve determinate and indeterminate beams."""
    start_time = perf_counter()
    supports = sorted(payload.supports, key=lambda s: s.position)
    N_s = len(supports)
    fixed_indices = [i for i, s in enumerate(supports) if s.type == "fixed"]
    N_f = len(fixed_indices)

    M = N_s + N_f + 2

    A = np.zeros((M, M), dtype=float)
    B = np.zeros(M, dtype=float)

    # Row 0: Vertical Equilibrium sum(R_i) = Total Downward applied loads
    for i in range(N_s):
        A[0, i] = 1.0

    total_p = sum(_vertical_component(load) for load in payload.point_loads)
    total_udl = 0.0
    for udl in payload.udls:
        q = udl.magnitude * _udl_sign(udl)
        span = udl.end - udl.start
        if span <= 0:
            continue
        if udl.shape == "uniform":
            total_udl += q * span
        else:  # triangular
            total_udl += 0.5 * q * span
    B[0] = total_p + total_udl

    # Row 1: Moment Equilibrium about x=0
    for i in range(N_s):
        A[1, i] = supports[i].position
    for j in range(N_f):
        A[1, N_s + j] = -1.0

    moment_p = sum(_vertical_component(load) * load.position for load in payload.point_loads)
    moment_udl = 0.0
    for udl in payload.udls:
        q = udl.magnitude * _udl_sign(udl)
        span = udl.end - udl.start
        if span <= 0:
            continue
        if udl.shape == "uniform":
            centroid = udl.start + span / 2.0
            moment_udl += q * span * centroid
        elif udl.shape == "triangular_increasing":
            centroid = udl.start + 2.0 * span / 3.0
            moment_udl += 0.5 * q * span * centroid
        elif udl.shape == "triangular_decreasing":
            centroid = udl.start + span / 3.0
            moment_udl += 0.5 * q * span * centroid
    moment_moments = sum(moment_load.magnitude * _moment_sign(moment_load.direction) for moment_load in payload.moment_loads)
    B[1] = moment_p + moment_udl + moment_moments

    # Rows 2 to 2 + N_s - 1: Deflection w(s_k) = 0
    for k, support in enumerate(supports):
        s_k = support.position
        for i in range(N_s):
            A[2 + k, i] = (1.0 / 6.0) * macaulay(np.array([s_k]), supports[i].position, 3)[0]
        for j, idx in enumerate(fixed_indices):
            A[2 + k, N_s + j] = (1.0 / 2.0) * macaulay(np.array([s_k]), supports[idx].position, 2)[0]
        A[2 + k, N_s + N_f] = s_k
        A[2 + k, N_s + N_f + 1] = 1.0
        _, _, w_val = get_applied_loads_at_x(s_k, payload)
        B[2 + k] = w_val

    # Rows 2 + N_s to 2 + N_s + N_f - 1: Rotation theta(f_k) = 0
    for j, idx in enumerate(fixed_indices):
        f_j = supports[idx].position
        for i in range(N_s):
            A[2 + N_s + j, i] = (1.0 / 2.0) * macaulay(np.array([f_j]), supports[i].position, 2)[0]
        for j_prime, idx_prime in enumerate(fixed_indices):
            A[2 + N_s + j, N_s + j_prime] = macaulay(np.array([f_j]), supports[idx_prime].position, 1)[0]
        A[2 + N_s + j, N_s + N_f] = 1.0
        _, theta_val, _ = get_applied_loads_at_x(f_j, payload)
        B[2 + N_s + j] = theta_val

    # Solve linear system
    try:
        X = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        raise ValueError("Kiriş dengesiz veya tanımsız mesnet yerleşimi mevcut.")

    reactions_vertical = X[0:N_s]
    reactions_moment = X[N_s : N_s + N_f]
    C1 = X[N_s + N_f]
    C2 = X[N_s + N_f + 1]

    # Construct resolved reactions
    resolved_reactions = []
    fixed_count = 0
    for i, support in enumerate(supports):
        vert = reactions_vertical[i]
        moment_val = 0.0
        if support.type == "fixed":
            moment_val = reactions_moment[fixed_count]
            fixed_count += 1
        resolved_reactions.append(
            SupportReaction(
                support_id=support.id,
                support_type=support.type,
                position=_format_float(support.position),
                vertical=_format_float(vert),
                axial=0.0,
                moment=_format_float(moment_val),
            )
        )

    # Axial reactions
    total_axial_load = sum(_axial_component(load) for load in payload.point_loads)
    axial_support_index = None
    for i, support in enumerate(supports):
        if support.type in ("pin", "fixed"):
            axial_support_index = i
            break

    for i, reaction in enumerate(resolved_reactions):
        if i == axial_support_index:
            reaction.axial = _format_float(-total_axial_load)
        else:
            reaction.axial = 0.0

    E = payload.elastic_modulus_gpa
    I = payload.moment_inertia_m4
    EI = E * 1e6 * I

    # Build refined axis
    supports_positions = [s.position for s in supports]
    x_axis = build_refined_axis(payload, supports_positions)

    discontinuity_positions = []
    for r in resolved_reactions:
        if abs(r.vertical) > ROOT_TOL:
            discontinuity_positions.append(r.position)
    for load in payload.point_loads:
        p_val = _vertical_component(load)
        if abs(p_val) > ROOT_TOL:
            discontinuity_positions.append(load.position)
    for m in payload.moment_loads:
        discontinuity_positions.append(m.position)

    if discontinuity_positions:
        x_axis_refined = []
        for x_val in x_axis:
            is_jump = any(math.isclose(x_val, pos, abs_tol=ROOT_TOL, rel_tol=0.0) for pos in discontinuity_positions)
            if is_jump and x_val > 0.0:
                left_eval = float(np.nextafter(x_val, -np.inf))
                x_axis_refined.append(left_eval)
            x_axis_refined.append(float(x_val))
        x_axis = np.array(sorted(list(set(x_axis_refined))), dtype=float)

    # Evaluate diagrams
    shear, moment, normal, theta, deflection_mm = evaluate_diagrams(x_axis, payload, resolved_reactions, C1, C2, EI)

    # Extremums
    moment_extrema = _compute_moment_extrema(payload, resolved_reactions, x_axis, shear)
    max_positive = moment_extrema.get("max_positive")
    min_negative = moment_extrema.get("min_negative")
    max_absolute = moment_extrema.get("max_absolute")

    max_deflection_val = float(np.max(np.abs(deflection_mm)))
    max_deflection_idx = np.argmax(np.abs(deflection_mm))
    max_deflection_pos = float(x_axis[max_deflection_idx])

    duration_ms = (perf_counter() - start_time) * 1000.0

    diagram_data = DiagramData(
        x=[_format_float(v) for v in x_axis],
        shear=[_format_float(v) for v in shear],
        moment=[_format_float(v) for v in moment],
        normal=[_format_float(v) for v in normal],
        deflection=[_format_float(v) for v in deflection_mm],
        rotation=[_format_float(v) for v in theta],
    )

    detailed_sol = generate_detailed_solution_steps(
        payload, resolved_reactions, C1, C2, E, I, max_deflection_val, max_deflection_pos
    )

    return SolveResponse(
        reactions=resolved_reactions,
        diagram=diagram_data,
        meta=SolveMeta(
            solve_time_ms=_format_float(duration_ms),
            validation_warnings=[],
            recommendation=MethodRecommendation(
                method="area",
                title="Alan Yontemi",
                reason="Standart cozum yontemi."
            ),
            max_positive_moment=_format_float(max_positive[1]) if max_positive else None,
            max_positive_position=_format_float(max_positive[0]) if max_positive else None,
            min_negative_moment=_format_float(min_negative[1]) if min_negative else None,
            min_negative_position=_format_float(min_negative[0]) if min_negative else None,
            max_absolute_moment=_format_float(max_absolute[1]) if max_absolute else None,
            max_absolute_position=_format_float(max_absolute[0]) if max_absolute else None,
            max_deflection=_format_float(max_deflection_val),
            max_deflection_position=_format_float(max_deflection_pos),
        ),
        detailed_solution=detailed_sol,
    )


def solve_beam(payload: SolveRequest) -> SolveResponse:
    """Solve beam using unified Macaulay solver."""
    return solve_beam_unified(payload)


def solve_cantilever_beam(payload: SolveRequest) -> SolveResponse:
    """Solve cantilever beam using unified Macaulay solver."""
    return solve_beam_unified(payload)



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
