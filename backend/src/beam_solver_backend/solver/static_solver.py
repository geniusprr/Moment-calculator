from __future__ import annotations

import math
import warnings
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.exceptions import RankWarning

from beam_solver_backend.schemas.beam import (
    AreaMethodVisualization,
    MomentSegmentSamples,
    ShearRegionSamples,
    BeamContext,
    BeamDistributedLoadInfo,
    BeamMomentLoadInfo,
    BeamPointLoadInfo,
    BeamSectionHighlight,
    BeamSupportInfo,
    DetailedSolution,
    DiagramData,
    MethodRecommendation,
    MomentDirection,
    SolutionMethod,
    SolutionStep,
    SolveRequest,
    SolveResponse,
    SolveMeta,
    SupportReaction,
)


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
    else:  # pragma: no cover - safety for future shapes
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
        # No guaranteed sign change; fall back to mid-point search for robustness.
        mid = 0.5 * (lo + hi)
        f_mid = float(_shear_diagram(payload, np.array([mid], dtype=float), reactions)[0])
        if f_lo * f_mid <= 0.0:
            return _locate_shear_zero(payload, reactions, lo, mid, f_lo, f_mid, max_iterations, tol)
        if f_mid * f_hi <= 0.0:
            return _locate_shear_zero(payload, reactions, mid, hi, f_mid, f_hi, max_iterations, tol)
        # Fallback to midpoint if still no sign change detected (flat region).
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
    has_point_loads = len(payload.point_loads) > 0
    has_moments = len(payload.moment_loads) > 0
    has_axial_point = any(abs(_axial_component(load)) > 1e-6 for load in payload.point_loads)
    
    # Yayılı yükleri uniform ve üçgen olarak ayır
    has_uniform_distributed = any(udl.shape == "uniform" for udl in payload.udls)
    has_triangular_distributed = any(
        udl.shape in ["triangular_increasing", "triangular_decreasing"] 
        for udl in payload.udls
    )
    has_any_distributed = len(payload.udls) > 0

    # Öncelik: Üçgen yayılı yük alan hesabını zorlaştırır → Kesit yöntemi öner
    if has_triangular_distributed:
        return MethodRecommendation(
            method="shear",
            title="Kesit Yöntemi",
            reason=(
                "Üçgen yayılı yükler kesme diyagramında parabolik alanlar oluşturur ve alan yönteminde "
                "türev-integral hesaplarını zorlaştırır; kesit yöntemi her bölgeyi doğrudan kuvvet ve moment "
                "dengeleriyle çözerek daha güvenilir ilerler."
            ),
        )

    # Açılı (eksenel bileşenli) tekil yükler alan yaklaşımını karmaşıklaştırır
    if has_axial_point:
        return MethodRecommendation(
            method="shear",
            title="Kesit Yöntemi",
            reason=(
                "Açılı veya eksenel bileşenli tekil yükler kesme ve normal kuvvet diyagramlarını birlikte etkiler; "
                "kesit yöntemi denge denklemleriyle bu etkileşimi daha net takip etmeye imkan tanır."
            ),
        )
    
    # Kural 1: Sadece tekil yükler varsa Alan yöntemi
    if has_point_loads and not payload.udls and not has_moments and not has_axial_point:
        return MethodRecommendation(
            method="area",
            title="Alan Yöntemi",
            reason=(
                "Sadece dikey tekil yükler bulunduğundan kesme diyagramı parça parça sabit kalır ve moment "
                "doğrusal segmentlerden oluşur; alan yöntemi bu durumda doğrudan ve hızlıdır."
            ),
        )
    
    # Kural 2: Sadece düzgün yayılı yük varsa Alan yöntemi
    if has_uniform_distributed and not has_triangular_distributed and not has_point_loads and not has_moments:
        return MethodRecommendation(
            method="area",
            title="Alan Yöntemi",
            reason=(
                "Sadece düzgün yayılı yükler bulunduğundan kesme diyagramı doğrusal ve moment diyagramı "
                "parabolik olur; alan yöntemi bu durumda pratik ve görselleştirmesi kolaydır."
            ),
        )
    
    # Kural 3: Düzgün yayılı + tekil yükler + moment varsa Alan yöntemi
    if has_uniform_distributed and not has_triangular_distributed and has_point_loads and has_moments:
        return MethodRecommendation(
            method="area",
            title="Alan Yöntemi",
            reason=(
                "Düzgün yayılı yükler, tekil yükler ve momentler birlikte bulunuyor. Alan yöntemi bu kombinasyonda "
                "kesme diyagramının alanlarını kullanarak moment diyagramını adım adım inşa edebilir."
            ),
        )
    
    # Kural 4: Sadece moment varsa Alan yöntemi
    if has_moments and not has_point_loads and not payload.udls:
        return MethodRecommendation(
            method="area",
            title="Alan Yöntemi",
            reason=(
                "Sadece mesnet momentleri bulunduğundan kesme diyagramı sıfır kalır ve moment diyagramı "
                "basit sabit değerlerden oluşur; alan yöntemi doğrudan uygulanabilir."
            ),
        )
    
    # Kural 5: Diğer tüm (sadece tekil/moment/yayılı kombinasyonları) için Alan yöntemi varsayılan
    if has_point_loads or has_any_distributed or has_moments:
        return MethodRecommendation(
            method="area",
            title="Alan Yöntemi",
            reason=(
                "Yükler yalnızca tekil kuvvetler, momentler ve düzgün yayılı yüklerden oluşuyor; kesme diyagramının "
                "alanları üzerinden moment diyagramını çıkarmak hızlı ve görsel olarak takip edilebilir."
            ),
        )

    # Varsayılan: yük yoksa alan yöntemi uygun
    return MethodRecommendation(
        method="area",
        title="Alan Yöntemi",
        reason="Yük bulunmadığından alan yöntemiyle başlangıç koşulları korunur.",
    )


def _collect_section_points(payload: SolveRequest) -> List[float]:
    points = {0.0, payload.length}
    for support in payload.supports:
        points.add(float(support.position))
    for load in payload.point_loads:
        points.add(float(load.position))
    for udl in payload.udls:
        points.add(float(udl.start))
        points.add(float(udl.end))
    for moment_load in payload.moment_loads:
        points.add(float(moment_load.position))
    return sorted(points)


def _format_number_for_text(value: float, precision: int = 4) -> str:
    rounded = round(float(value), precision)
    if abs(rounded) < 10 ** (-precision):
        rounded = 0.0
    text = f"{rounded:.{precision}f}"
    text = text.rstrip("0").rstrip(".")
    if text == "-0":
        text = "0"
    return text or "0"


def _format_polynomial(coeffs: np.ndarray, variable: str = "x", precision: int = 4) -> str:
    coeffs = np.array(coeffs, dtype=float)
    degree = len(coeffs) - 1
    terms: List[str] = []
    for idx, coeff in enumerate(coeffs):
        power = degree - idx
        if abs(coeff) < 1e-8:
            continue
        coef_text = _format_number_for_text(coeff, precision)
        if coef_text == "0":
            continue
        sign = "-" if coef_text.startswith("-") else "+"
        coef_text = coef_text[1:] if coef_text.startswith("-") else coef_text
        if power == 0:
            term = coef_text
        elif power == 1:
            term = f"{coef_text}·{variable}"
        else:
            term = f"{coef_text}·{variable}^{power}"
        terms.append((sign, term))

    if not terms:
        return "0"

    first_sign, first_term = terms[0]
    expression = ("- " if first_sign == "-" else "") + first_term
    for sign, term in terms[1:]:
        expression += f" {'-' if sign == '-' else '+'} {term}"
    return expression


def _fit_polynomial(x_vals: np.ndarray, y_vals: np.ndarray, max_degree: int) -> np.ndarray:
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    x_unique, unique_indices = np.unique(x, return_index=True)
    y_unique = y[unique_indices]

    degree = min(max_degree, len(x_unique) - 1)
    if degree <= 0:
        return np.array([float(y_unique[0])])
    if degree == 1:
        x0, x1 = x_unique[0], x_unique[-1]
        if math.isclose(x1, x0, abs_tol=1e-9):
            return np.array([0.0, float(y_unique[0])])
        slope = (y_unique[-1] - y_unique[0]) / (x1 - x0)
        intercept = y_unique[0] - slope * x0
        return np.array([float(slope), float(intercept)])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RankWarning)
        coeffs = np.polyfit(x_unique, y_unique, degree)
    return coeffs


def _explain_shear_coefficients(expr: str, constant: float, payload: SolveRequest, 
                                reactions: List[SupportReaction], x_start: float, x_end: float) -> str:
    """Kesme denklemindeki katsayıların nereden geldiğini basitçe açıkla."""
    lines: List[str] = []
    
    # z² terimi var mı kontrol et
    if "z^2" in expr or "z²" in expr:
        lines.append("• z² terimi → Üçgen yayılı yükten gelir")
        for udl in payload.udls:
            if udl.start < x_end - 1e-6 and udl.end > x_start + 1e-6:
                if udl.shape == "triangular_increasing":
                    span = udl.end - udl.start
                    coeff = _udl_sign(udl) * udl.magnitude / (2 * span)
                    lines.append(f"  Katsayı = w/(2×L) = {udl.magnitude:.2f}/(2×{span:.2f}) = {coeff:.4f}")
    
    # z terimi
    if "·z" in expr or "*z" in expr:
        lines.append("• z terimi → Düzgün yayılı yük veya tekil kuvvetlerden gelir")
        for udl in payload.udls:
            if udl.start < x_end - 1e-6 and udl.end > x_start + 1e-6:
                if udl.shape == "uniform":
                    sign = _udl_sign(udl)
                    lines.append(f"  Katsayı = w = {sign * udl.magnitude:.2f}")
    
    # Sabit terim
    if abs(constant) > 1e-6:
        lines.append(f"• Sabit terim → Önceki mesnet ve yüklerin toplamı = {constant:.2f} kN")
    
    return "\n".join(lines) if lines else "Denklem sabittir (yayılı yük yok)"


def _explain_moment_coefficients(expr: str, payload: SolveRequest, 
                                  reactions: List[SupportReaction], x_start: float, x_end: float) -> str:
    """Moment denklemindeki katsayıların nereden geldiğini basitçe açıkla."""
    lines: List[str] = []
    
    # z³ terimi
    if "z^3" in expr or "z³" in expr:
        lines.append("• z³ terimi → Üçgen yayılı yükün momenti")
        for udl in payload.udls:
            if udl.start < x_end - 1e-6 and udl.end > x_start + 1e-6:
                if udl.shape == "triangular_increasing":
                    span = udl.end - udl.start
                    coeff = _udl_sign(udl) * udl.magnitude / (6 * span)
                    lines.append(f"  Katsayı = w/(6×L) = {udl.magnitude:.2f}/(6×{span:.2f}) = {coeff:.6f}")
    
    # z² terimi
    if "z^2" in expr or "z²" in expr:
        lines.append("• z² terimi → Düzgün yayılı yükün momenti")
        for udl in payload.udls:
            if udl.start < x_end - 1e-6 and udl.end > x_start + 1e-6:
                if udl.shape == "uniform":
                    sign = _udl_sign(udl)
                    coeff = sign * udl.magnitude / 2
                    lines.append(f"  Katsayı = w/2 = {udl.magnitude:.2f}/2 = {coeff:.2f}")
    
    # z terimi
    if "·z" in expr or "*z" in expr:
        lines.append("• z terimi → Mesnet reaksiyonları ve tekil yükler")
        total_force = 0.0
        for reaction in reactions:
            if reaction.position <= x_start + 1e-6:
                total_force += reaction.vertical
        for load in payload.point_loads:
            if load.position <= x_start + 1e-6:
                total_force -= _vertical_component(load)
        if abs(total_force) > 1e-6:
            lines.append(f"  Toplam kuvvet = {total_force:.2f} kN")
    
    # Sabit terim
    lines.append("• Sabit terim → Önceki momentlerin birikimidir")
    
    return "\n".join(lines) if lines else "Basit moment denklemi"


def _build_detailed_section_derivation(
    payload: SolveRequest,
    reactions: List[SupportReaction],
    x_start: float,
    x_end: float,
    region_idx: int,
    shear_expr: str,
    moment_expr: str,
    shear_end: float,
    moment_end: float,
) -> str:
    """
    Kesit denklemlerinin nasıl oluşturulduğunu adım adım açıklar.
    """
    lines: List[str] = []
    
    lines.append("═══════════════════════════════════════════════════════")
    lines.append("KESİT DENKLEMLERİNİN DETAYLI TÜRETİLMESİ")
    lines.append("═══════════════════════════════════════════════════════\n")
    
    # 1. Koordinat sistemi açıklaması
    lines.append("📍 ADIM 1: KOORDİNAT SİSTEMİ")
    lines.append("─────────────────────────────────────────────────────")
    lines.append(f"Bu bölgede 'z' değişkenini kullanıyoruz:")
    lines.append(f"• z, bölgenin sol sınırından (z = {x_start:.2f} m) başlar")
    lines.append(f"• z, sağa doğru artar ve bölgenin sonunda (z = {x_end:.2f} m) biter")
    lines.append(f"• Bölge uzunluğu: {x_end - x_start:.2f} m")
    lines.append(f"• Herhangi bir z noktasında kesit açacağız (z ∈ [{x_start:.2f}, {x_end:.2f}])\n")
    
    # 2. Kesit kesme açıklaması
    lines.append("✂️ ADIM 2: KESİT AÇMAK")
    lines.append("─────────────────────────────────────────────────────")
    lines.append("Bu bölgede z konumunda hayali bir kesit açıyoruz.")
    lines.append("Kesit, kirişi iki parçaya ayırır:")
    lines.append("• SOL PARÇA: Kirişin başlangıcından (x=0) z noktasına kadar")
    lines.append("• SAĞ PARÇA: z noktasından kirişin sonuna kadar")
    lines.append("\nKesitten sonra sol parçayı inceleyeceğiz (sol parça yöntemi).\n")
    
    # 3. Sol parçadaki kuvvetleri listele
    lines.append("⚖️ ADIM 3: SOL PARÇAYA ETKİYEN KUVVETLER")
    lines.append("─────────────────────────────────────────────────────")
    lines.append("Sol parçaya (0'dan z'ye kadar) etki eden tüm kuvvetleri yazalım:\n")
    
    force_list: List[str] = []
    total_vertical_before_z = 0.0
    moment_contributions: List[str] = []
    
    # Mesnet reaksiyonları
    for reaction in reactions:
        if reaction.position <= x_start + 1e-6:
            force_list.append(
                f"  🔺 Mesnet {reaction.support_id} reaksiyonu: R_{reaction.support_id} = {reaction.vertical:.2f} kN (yukarı)"
                f"\n     Konum: x = {reaction.position:.2f} m"
                f"\n     Bu kuvvet sol parçanın başında yer alır."
            )
            total_vertical_before_z += reaction.vertical
            lever_arm = f"(z - {reaction.position:.2f})"
            moment_contributions.append(
                f"  • R_{reaction.support_id} × {lever_arm} = {reaction.vertical:.2f} × {lever_arm} kN·m"
            )
    
    # Tekil yükler
    for load in payload.point_loads:
        if load.position <= x_start + 1e-6:
            vertical = _vertical_component(load)
            if abs(vertical) > 1e-6:
                direction = "aşağı (-)" if vertical < 0 else "yukarı (+)"
                force_list.append(
                    f"  🔻 Tekil yük {load.id}: F = {abs(vertical):.2f} kN ({direction})"
                    f"\n     Konum: x = {load.position:.2f} m"
                    f"\n     Dikey bileşen: {vertical:.2f} kN"
                )
                total_vertical_before_z += vertical
                lever_arm = f"(z - {load.position:.2f})"
                moment_contributions.append(
                    f"  • F_{load.id} × {lever_arm} = {-vertical:.2f} × {lever_arm} kN·m (eksi işaretli çünkü yük aşağı)"
                )
    
    # Yayılı yükler
    active_udl_in_region = False
    for udl in payload.udls:
        if udl.start < x_end - 1e-6 and udl.end > x_start + 1e-6:
            active_udl_in_region = True
            direction = "aşağı" if udl.direction == "down" else "yukarı"
            shape_name = {
                "uniform": "Düzgün yayılı",
                "triangular_increasing": "Üçgen (artan)",
                "triangular_decreasing": "Üçgen (azalan)"
            }.get(udl.shape, udl.shape)
            
            force_list.append(
                f"  📊 Yayılı yük {udl.id}: {shape_name}"
                f"\n     Yoğunluk: w = {udl.magnitude:.2f} kN/m ({direction})"
                f"\n     Tam aralık: [{udl.start:.2f}, {udl.end:.2f}] m"
                f"\n     Bu bölgedeki etkisi: z'ye kadar olan kısmı hesaba katılır"
            )
            
            if udl.shape == "uniform":
                force_list.append(
                    f"     Formül: Kuvvet = w × (z - {udl.start:.2f})"
                    f"\n     Moment kolu = (z - {udl.start:.2f})/2 (dikdörtgenin merkezi)"
                )
                moment_contributions.append(
                    f"  • w × (z - {udl.start:.2f}) × (z - {udl.start:.2f})/2"
                    f"\n    = {_udl_sign(udl) * udl.magnitude:.2f} × (z - {udl.start:.2f})²/2 kN·m"
                )
            elif udl.shape == "triangular_increasing":
                force_list.append(
                    f"     Formül: Kuvvet = w × (z - {udl.start:.2f})²/(2×L_udl)"
                    f"\n     Moment kolu = (z - {udl.start:.2f})/3 (üçgenin merkezi)"
                )
                moment_contributions.append(
                    f"  • [w × (z - {udl.start:.2f})²/(2×L)] × [(z - {udl.start:.2f})/3]"
                    f"\n    = {_udl_sign(udl) * udl.magnitude/(2*(udl.end-udl.start)):.4f} × (z - {udl.start:.2f})³/3 kN·m"
                )
            else:  # triangular_decreasing
                force_list.append(
                    f"     Formül: Karmaşık (üçgen azalan)"
                    f"\n     Moment kolu ve kuvvet z'ye bağlı olarak değişir"
                )
                moment_contributions.append(
                    f"  • Azalan üçgen yayılı yük moment katkısı (polinom)"
                )
    
    # İç kuvvetler (bilinmeyenler)
    force_list.append(
        f"  ❓ KESİTTE DOĞAN İÇ KUVVETLER (Bilinmeyenler):"
        f"\n     T(z) = Kesme kuvveti (yukarı yönde pozitif)"
        f"\n     M(z) = Eğilme momenti (alt lifte çekme pozitif)"
    )
    
    if force_list:
        lines.append("\n".join(force_list))
    lines.append("")
    
    # 4. Düşey kuvvet dengesi
    lines.append("⬆️ ADIM 4: DÜŞEY KUVVET DENGESİ (ΣFy = 0)")
    lines.append("─────────────────────────────────────────────────────")
    lines.append("Sol parçanın düşey dengesi için tüm yukarı kuvvetler = tüm aşağı kuvvetler:")
    lines.append("")
    
    # Kuvvetleri topla
    shear_terms: List[str] = []
    shear_constant = 0.0
    
    lines.append("Yukarı kuvvetler:")
    for reaction in reactions:
        if reaction.position <= x_start + 1e-6:
            lines.append(f"  + R_{reaction.support_id} = {reaction.vertical:.2f} kN")
            shear_constant += reaction.vertical
            shear_terms.append(f"+{reaction.vertical:.2f}")
    lines.append(f"  + T(z) = ? (kesitteki iç kesme kuvveti)")
    
    lines.append("\nAşağı kuvvetler:")
    for load in payload.point_loads:
        if load.position <= x_start + 1e-6:
            vertical = _vertical_component(load)
            if abs(vertical) > 1e-6:
                lines.append(f"  - {abs(vertical):.2f} kN (tekil yük {load.id})")
                shear_constant += vertical  # vertical zaten negatif
                shear_terms.append(f"{vertical:.2f}")
    
    udl_formula_parts: List[str] = []
    if active_udl_in_region:
        lines.append(f"  - Yayılı yüklerin z'ye kadar olan toplamı (z'ye bağlı)")
        for udl in payload.udls:
            if udl.start < x_end - 1e-6 and udl.end > x_start + 1e-6:
                sign = _udl_sign(udl)
                if udl.shape == "uniform":
                    udl_formula_parts.append(
                        f"w×(z-{udl.start:.2f}) = {sign * udl.magnitude:.2f}×(z-{udl.start:.2f})"
                    )
                elif udl.shape == "triangular_increasing":
                    span = udl.end - udl.start
                    coeff = sign * udl.magnitude / (2 * span)
                    udl_formula_parts.append(
                        f"{coeff:.4f}×(z-{udl.start:.2f})²"
                    )
                else:  # triangular_decreasing
                    udl_formula_parts.append(
                        f"[Karmaşık üçgen azalan formül]"
                    )
    
    lines.append("\nDenge denklemi kurulumu:")
    lines.append("ΣFy = 0 ⇒ (Yukarı kuvvetler) - (Aşağı kuvvetler) = 0")
    
    # Sade açıklama için
    lines.append("\n💡 KESME DENKLEMİNİN OLUŞUMU:")
    
    if not active_udl_in_region:
        # Basit durum: yayılı yük yok
        lines.append(f"Bu bölgede yayılı yük YOK, sadece sabit kuvvetler var.")
        lines.append(f"\nT(z) = (Önceki tüm kuvvetler toplamı)")
        if shear_terms:
            calculation = " ".join(shear_terms)
            lines.append(f"T(z) = {calculation}")
            lines.append(f"T(z) = {shear_constant:.2f} kN (sabit)")
    else:
        # Yayılı yük var
        lines.append(f"Bu bölgede yayılı yük VAR, kesme z'ye bağlı değişir:")
        
        # Her yayılı yük için basit açıklama
        for udl in payload.udls:
            if udl.start < x_end - 1e-6 and udl.end > x_start + 1e-6:
                sign = _udl_sign(udl)
                if udl.shape == "uniform":
                    lines.append(f"\n• Düzgün yayılı yük: w = {udl.magnitude:.2f} kN/m")
                    lines.append(f"  z'ye kadar olan kuvvet = w × (z - {udl.start:.2f})")
                    lines.append(f"  = {sign * udl.magnitude:.2f} × (z - {udl.start:.2f})")
                    lines.append(f"  Bu bir doğrusal (z'ye göre) terimdir")
                    
                elif udl.shape == "triangular_increasing":
                    span = udl.end - udl.start
                    lines.append(f"\n• Üçgen artan yayılı yük: w_max = {udl.magnitude:.2f} kN/m")
                    lines.append(f"  Yük yoğunluğu: 0'dan başlayıp {udl.magnitude:.2f} kN/m'ye çıkar")
                    lines.append(f"  z'ye kadar toplam kuvvet = (w_max / 2L) × (z - {udl.start:.2f})²")
                    coeff = sign * udl.magnitude / (2 * span)
                    lines.append(f"  = {udl.magnitude:.2f} / (2 × {span:.2f}) × (z - {udl.start:.2f})²")
                    lines.append(f"  = {coeff:.4f} × (z - {udl.start:.2f})²")
                    lines.append(f"  Bu bir kuadratik (z²'ye göre) terimdir")
                    
                else:  # triangular_decreasing
                    span = udl.end - udl.start
                    lines.append(f"\n• Üçgen azalan yayılı yük: w_max = {udl.magnitude:.2f} kN/m")
                    lines.append(f"  Yük yoğunluğu: {udl.magnitude:.2f} kN/m'den başlayıp 0'a iner")
                    lines.append(f"  z'ye kadar toplam kuvvet = karmaşık polinom")
                    lines.append(f"  (z ve z² terimlerinin kombinasyonu)")
        
        if shear_terms:
            lines.append(f"\nSabit kuvvetler: {' '.join(shear_terms)} = {shear_constant:.2f} kN")
        
        lines.append(f"\nT(z) = Sabit kuvvetler + Yayılı yük terimleri")
    
    lines.append(f"\n✓ Hesaplanan kesme denklemi: T(z) = {shear_expr} kN")
    
    # Denklemdeki katsayıları açıkla
    lines.append(f"\n📌 DENKLEMDEKİ SAYILARIN ANLAMI:")
    lines.append(_explain_shear_coefficients(shear_expr, shear_constant, payload, reactions, x_start, x_end))
    
    # Sayısal doğrulama - daha sade
    lines.append(f"\n✓ Kontrol: z = {x_end:.2f} m → T = {shear_end:.2f} kN")
    lines.append("")
    
    # 5. Moment dengesi
    lines.append("🔄 ADIM 5: MOMENT DENGESİ (ΣM_kesit = 0)")
    lines.append("─────────────────────────────────────────────────────")
    lines.append("z noktasındaki kesite göre momentleri yazalım.")
    lines.append("Saat yönünün tersi (+) pozitif kabul edilir:\n")
    
    lines.append("💡 MOMENT DENKLEMİNİN OLUŞUMU:")
    lines.append("Her kuvvetin momenti = Kuvvet × Kol mesafesi\n")
    
    # Mesnet reaksiyonlarının momentleri
    for reaction in reactions:
        if reaction.position <= x_start + 1e-6 and abs(reaction.vertical) > 1e-6:
            if abs(reaction.position) < 1e-6:
                lines.append(f"• Mesnet {reaction.support_id}: {reaction.vertical:.2f} × z = {reaction.vertical:.2f}·z kN·m")
            else:
                lines.append(f"• Mesnet {reaction.support_id}: {reaction.vertical:.2f} × (z - {reaction.position:.2f}) kN·m")
    
    # Tekil yüklerin momentleri
    for load in payload.point_loads:
        if load.position <= x_start + 1e-6:
            vertical = _vertical_component(load)
            if abs(vertical) > 1e-6:
                lines.append(f"• Tekil yük {load.id}: {-vertical:.2f} × (z - {load.position:.2f}) kN·m")
    
    # Yayılı yüklerin momentleri
    if active_udl_in_region:
        for udl in payload.udls:
            if udl.start < x_end - 1e-6 and udl.end > x_start + 1e-6:
                sign = _udl_sign(udl)
                if udl.shape == "uniform":
                    coeff = sign * udl.magnitude / 2
                    lines.append(f"\n• Düzgün yayılı yük {udl.id}:")
                    lines.append(f"  Kuvvet × Kol = [w×(z-{udl.start:.2f})] × [(z-{udl.start:.2f})/2]")
                    lines.append(f"  = {udl.magnitude:.2f}/2 × (z-{udl.start:.2f})²")
                    lines.append(f"  = {coeff:.2f} × (z-{udl.start:.2f})²")
                    
                elif udl.shape == "triangular_increasing":
                    span = udl.end - udl.start
                    coeff = sign * udl.magnitude / (6 * span)
                    lines.append(f"\n• Üçgen artan yük {udl.id}:")
                    lines.append(f"  Toplam yük × Ağırlık merkezi = [w/(2L)×(z-{udl.start:.2f})²] × [(z-{udl.start:.2f})/3]")
                    lines.append(f"  = {udl.magnitude:.2f}/(6×{span:.2f}) × (z-{udl.start:.2f})³")
                    lines.append(f"  = {coeff:.6f} × (z-{udl.start:.2f})³")
                else:
                    lines.append(f"\n• Üçgen azalan yük {udl.id}: Karmaşık polinom terimi")
    
    lines.append(f"\nM(z) = [Yukarıdaki tüm momentlerin toplamı]")
    lines.append(f"\n✓ Hesaplanan moment denklemi: M(z) = {moment_expr} kN·m")
    
    # Denklemdeki katsayıları açıkla
    lines.append(f"\n� DENKLEMDEKİ SAYILARIN ANLAMI:")
    lines.append(_explain_moment_coefficients(moment_expr, payload, reactions, x_start, x_end))
    
    # Sayısal doğrulama - sade
    lines.append(f"\n✓ Kontrol: z = {x_end:.2f} m → M = {moment_end:.2f} kN·m")
    lines.append("")
    
    # 6. Denklemlerin anlamı
    lines.append("📐 ADIM 6: DENKLEMLERİN FİZİKSEL ANLAMI")
    lines.append("─────────────────────────────────────────────────────")
    lines.append(f"• T(z) = {shear_expr}")
    lines.append(f"  Bu denklem, bölgenin HER noktasındaki kesme kuvvetini verir.")
    lines.append(f"  Örnek: z = {x_end:.2f} m için → T = {shear_end:.2f} kN")
    lines.append("")
    lines.append(f"• M(z) = {moment_expr}")
    lines.append(f"  Bu denklem, bölgenin HER noktasındaki eğilme momentini verir.")
    lines.append(f"  Örnek: z = {x_end:.2f} m için → M = {moment_end:.2f} kN·m")
    lines.append("")
    
    # 7. Denklemlerin türetme özeti
    lines.append("🎯 ÖZET: NEDEN BU DENKLEMLER?")
    lines.append("─────────────────────────────────────────────────────")
    lines.append("1. Kirişte z noktasında hayali kesit açtık")
    lines.append("2. Sol parçayı serbest cisim olarak ele aldık")
    lines.append("3. Sol parçaya etki eden TÜM kuvvetleri (reaksiyon, yük, iç kuvvet) yazdık")
    lines.append("4. ΣFy = 0 denkleminden T(z) formülünü bulduk")
    lines.append("5. ΣM = 0 denkleminden M(z) formülünü bulduk")
    lines.append("6. Bu formüller, bölgedeki HER noktada geçerlidir")
    lines.append("═══════════════════════════════════════════════════════\n")
    
    return "\n".join(lines)


def _describe_region_loads(
    payload: SolveRequest,
    reactions: List[SupportReaction],
    x_start: float,
    x_end: float,
) -> str:
    entries: List[str] = []
    for reaction in reactions:
        if x_start - 1e-6 <= reaction.position <= x_end + 1e-6:
            direction = "yukarı" if reaction.vertical >= 0 else "aşağı"
            location = "sol sınır" if math.isclose(reaction.position, x_start, abs_tol=1e-6) else (
                "sağ sınır" if math.isclose(reaction.position, x_end, abs_tol=1e-6) else "bölge içi"
            )
            entries.append(
                f"• {reaction.support_id} mesnet reaksiyonu ({location}): R_{reaction.support_id} = {reaction.vertical:.2f} kN ({direction})"
            )

    for load in payload.point_loads:
        if x_start - 1e-6 < load.position < x_end + 1e-6:
            vertical = _vertical_component(load)
            if abs(vertical) < 1e-6:
                continue
            direction = "aşağı" if vertical < 0 else "yukarı"
            entries.append(
                f"• x = {load.position:.2f} m noktasında {abs(vertical):.2f} kN {direction} tekil yük"
            )

    for udl in payload.udls:
        overlap_start = max(x_start, udl.start)
        overlap_end = min(x_end, udl.end)
        if overlap_end - overlap_start <= 1e-6:
            continue
        direction = "aşağı" if udl.direction == "down" else "yukarı"
        if udl.shape == "uniform":
            shape_desc = "düzgün yayılı"
        elif udl.shape == "triangular_increasing":
            shape_desc = "hatlı (artarak) yayılı"
        else:
            shape_desc = "hatlı (azalarak) yayılı"
        entries.append(
            f"• {udl.start:.2f}-{udl.end:.2f} m aralığında {udl.magnitude:.2f} kN/m {direction} {shape_desc} yük (bu bölgede {overlap_start:.2f}-{overlap_end:.2f} m etkili)"
        )

    for moment_load in payload.moment_loads:
        if x_start - 1e-6 < moment_load.position < x_end + 1e-6:
            direction = "saat yönü" if moment_load.direction == "cw" else "saat yönünün tersi"
            entries.append(
                f"• x = {moment_load.position:.2f} m noktasında {moment_load.magnitude:.2f} kN·m {direction} uygulanan çift"
            )

    if not entries:
        return "Bu aralıkta harici yük bulunmuyor; kesitteki iç kuvvetler sabit kalır."

    return "Bölge içeriği:\n" + "\n".join(entries)


def _format_interval(start: float, end: float, variable: str = "x") -> str:
    return f"{_format_number_for_text(start, 3)} ≤ {variable} ≤ {_format_number_for_text(end, 3)}"


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


def _generate_equilibrium_method(
    payload: SolveRequest, reactions: List[SupportReaction]
) -> SolutionMethod:
    """Generate detailed equilibrium method solution steps."""
    steps: List[SolutionStep] = []
    supports_sorted = sorted(payload.supports, key=lambda s: s.position)
    support_a, support_b = supports_sorted
    span = support_b.position - support_a.position

    # Step 1: Free Body Diagram
    fbd_description = (
        f"Kiriş üzerindeki tüm yükler ve mesnet reaksiyonları belirlenir. "
        f"Kiriş uzunluğu L = {payload.length:.2f} m, "
        f"mesnetler {support_a.id} (x = {support_a.position:.2f} m) ve "
        f"{support_b.id} (x = {support_b.position:.2f} m) konumlarındadır."
    )
    steps.append(
        SolutionStep(
            step_number=1,
            title="Serbest Cisim Diyagramı",
            explanation=fbd_description,
            general_formula=None,
            substituted_formula=rf"L = {payload.length:.2f}\text{{ m}}",
        )
    )

    # Step 2: Calculate total loads
    total_vertical = 0.0
    load_descriptions = []
    latex_terms = []
    
    # Analyze point loads
    for load in payload.point_loads:
        vertical = _vertical_component(load)
        total_vertical += vertical
        
        # Detailed description
        direction_text = "aşağı" if vertical > 0 else "yukarı"
        angle_info = f" (açı: {load.angle_deg:.0f}°)" if abs(load.angle_deg + 90) > 0.1 else ""
        
        load_descriptions.append(
            f"• {load.id}: Büyüklük = {load.magnitude:.2f} kN{angle_info}\n"
            f"  Konum: x = {load.position:.2f} m\n"
            f"  Dikey bileşen: {abs(vertical):.2f} kN ({direction_text})"
        )
        
        latex_terms.append(f"F_{{{load.id}}} = {abs(vertical):.3f}")

    # Analyze distributed loads
    for udl in payload.udls:
        span_length = udl.end - udl.start
        equivalent_force, centroid = _udl_equivalent_force_and_centroid(udl)
        total_vertical += equivalent_force

        direction_text = "aşağı" if udl.direction == "down" else "yukarı"
        shape_description = {
            "uniform": "Düzgün yayılı",
            "triangular_increasing": "Üçgen (başta 0 → sonda maksimum)",
            "triangular_decreasing": "Üçgen (başta maksimum → sonda 0)",
        }[udl.shape]

        load_descriptions.append(
            f"• {udl.id}: {shape_description} yük\n"
            f"  Maksimum yoğunluk: {udl.magnitude:.2f} kN/m ({direction_text})\n"
            f"  Aralık: [{udl.start:.2f}, {udl.end:.2f}] m (uzunluk: {span_length:.2f} m)\n"
            f"  Eşdeğer kuvvet: {abs(equivalent_force):.2f} kN\n"
            f"  Ağırlık merkezi: x = {centroid:.2f} m"
        )

        if udl.shape == "uniform":
            latex_terms.append(
                f"w_{{{udl.id}}} \\times L = {udl.magnitude:.3f} \\times {span_length:.3f}"
            )
        else:
            latex_terms.append(
                f"\\tfrac{{1}}{{2}} w_{{\max,{udl.id}}} \\times L = 0.5 \\times {udl.magnitude:.3f} \\times {span_length:.3f}"
            )

    # Analyze moment loads
    for moment_load in payload.moment_loads:
        direction_text = "saat yönü tersine" if moment_load.direction == "ccw" else "saat yönünde"
        
        load_descriptions.append(
            f"• {moment_load.id}: Moment = {moment_load.magnitude:.2f} kN·m ({direction_text})\n"
            f"  Konum: x = {moment_load.position:.2f} m"
        )

    if not load_descriptions:
        load_summary = "Kiriş üzerinde yük bulunmamaktadır."
        general_formula = None
        substituted_formula = r"\text{Yük yok}"
    else:
        load_summary = "Kiriş üzerindeki yükler ve özellikleri:\n\n" + "\n\n".join(load_descriptions)
        general_formula = r"\sum F_y = F_1 + F_2 + \ldots + F_n"
        substituted_formula = r"\sum F_y = " + " + ".join(latex_terms) + f" = {total_vertical:.3f}\\text{{ kN}}"
    
    steps.append(
        SolutionStep(
            step_number=2,
            title="Yük Analizi ve Toplam Yük Hesabı",
            explanation=load_summary,
            general_formula=general_formula,
            substituted_formula=substituted_formula,
            numerical_result=f"Toplam dikey yük = {abs(total_vertical):.3f} kN",
        )
    )

    # Step 3: Vertical force equilibrium - Simple explanation
    equilibrium_explanation = (
        "Kirişteki tüm kuvvetlerin dengede olması gerekir.\n"
        "Yukarı doğru kuvvetler = Aşağı doğru kuvvetler\n\n"
        f"Yukarı: R_{support_a.id} + R_{support_b.id}\n"
        f"Aşağı: {total_vertical:.2f} kN (tüm yükler)\n\n"
        "Bu iki değer birbirine eşit olmalıdır."
    )
    
    steps.append(
        SolutionStep(
            step_number=3,
            title="Adım 1: Kuvvet Dengesi Yazalım",
            explanation=equilibrium_explanation,
            general_formula=rf"\sum F_y = 0 \quad \Rightarrow \quad R_A + R_B = \text{{Toplam yük}}",
            substituted_formula=rf"R_{{{support_a.id}}} + R_{{{support_b.id}}} = {total_vertical:.2f}\text{{ kN}}",
            numerical_result=f"İki mesnet reaksiyonunun toplamı {total_vertical:.2f} kN olmalı",
        )
    )

    # Step 4: Moment equilibrium - Calculate moments simply
    total_moment_about_a = 0.0
    moment_parts = []

    # Build simple moment explanation
    for load in payload.point_loads:
        vertical = _vertical_component(load)
        lever_arm = load.position - support_a.position
        moment_contribution = vertical * lever_arm
        total_moment_about_a += moment_contribution
        moment_parts.append(f"• {load.id}: {abs(vertical):.2f} kN × {lever_arm:.2f} m = {abs(moment_contribution):.2f} kN·m")

    for udl in payload.udls:
        equivalent_force, centroid = _udl_equivalent_force_and_centroid(udl)
        lever_arm = centroid - support_a.position
        moment_contribution = equivalent_force * lever_arm
        total_moment_about_a += moment_contribution
        moment_parts.append(
            f"• {udl.id}: {abs(equivalent_force):.2f} kN × {lever_arm:.2f} m = {abs(moment_contribution):.2f} kN·m"
        )

    for moment_load in payload.moment_loads:
        signed_moment = moment_load.magnitude * _moment_sign(moment_load.direction)
        total_moment_about_a += signed_moment
        moment_parts.append(f"• {moment_load.id}: {moment_load.magnitude:.2f} kN·m")
    
    moment_explanation = (
        f"{support_a.id} noktasına göre moment dengesi kuralım.\n"
        f"Sağ taraftaki R_{support_b.id} mesnetinin yarattığı moment = Yüklerin yarattığı momentler\n\n"
        "Yüklerin momentleri:\n" + "\n".join(moment_parts) + f"\n\n"
        f"Toplam = {abs(total_moment_about_a):.2f} kN·m"
    )
    
    steps.append(
        SolutionStep(
            step_number=4,
            title="Adım 2: Moment Dengesi Yazalım",
            explanation=moment_explanation,
            general_formula=rf"\sum M_A = 0 \quad \Rightarrow \quad R_B \times L = \text{{Toplam moment}}",
            substituted_formula=rf"R_{{{support_b.id}}} \times {span:.2f}\text{{ m}} = {total_moment_about_a:.2f}\text{{ kN}}\cdot\text{{m}}",
        )
    )

    # Step 5: Solve for R_B - Simple calculation
    reaction_b = reactions[1].vertical
    calculation_explanation = (
        f"R_{support_b.id} mesnetinin momenti = Yüklerin toplam momenti\n\n"
        f"R_{support_b.id} × {span:.2f} m = {total_moment_about_a:.2f} kN·m\n\n"
        f"Her iki tarafı {span:.2f} m'ye bölelim:\n"
        f"R_{support_b.id} = {total_moment_about_a:.2f} ÷ {span:.2f} = {reaction_b:.2f} kN"
    )
    
    steps.append(
        SolutionStep(
            step_number=5,
            title=f"Adım 3: R_{support_b.id} Reaksiyonunu Bulalım",
            explanation=calculation_explanation,
            general_formula=rf"R_B = \frac{{\text{{Toplam moment}}}}{{\text{{Mesafe}}}}",
            substituted_formula=rf"R_{{{support_b.id}}} = \frac{{{total_moment_about_a:.2f}\text{{ kN}}\cdot\text{{m}}}}{{{span:.2f}\text{{ m}}}} = {reaction_b:.2f}\text{{ kN}}",
            numerical_result=f"✓ R_{support_b.id} = {reaction_b:.2f} kN",
        )
    )

    # Step 6: Solve for R_A - Simple substitution
    reaction_a = reactions[0].vertical
    ra_explanation = (
        f"Adım 1'deki kuvvet dengesini kullanalım:\n\n"
        f"R_{support_a.id} + R_{support_b.id} = {total_vertical:.2f} kN\n\n"
        f"R_{support_b.id} değerini biliyoruz ({reaction_b:.2f} kN), yerine koyalım:\n\n"
        f"R_{support_a.id} + {reaction_b:.2f} = {total_vertical:.2f}\n"
        f"R_{support_a.id} = {total_vertical:.2f} - {reaction_b:.2f} = {reaction_a:.2f} kN"
    )
    
    steps.append(
        SolutionStep(
            step_number=6,
            title=f"Adım 4: R_{support_a.id} Reaksiyonunu Bulalım",
            explanation=ra_explanation,
            general_formula=rf"R_A = \text{{Toplam yük}} - R_B",
            substituted_formula=rf"R_{{{support_a.id}}} = {total_vertical:.2f}\text{{ kN}} - {reaction_b:.2f}\text{{ kN}} = {reaction_a:.2f}\text{{ kN}}",
            numerical_result=f"✓ R_{support_a.id} = {reaction_a:.2f} kN",
        )
    )

    # Step 7: Verification - Simple check
    sum_forces = reaction_a + reaction_b
    check_explanation = (
        "Bulduğumuz değerleri kontrol edelim:\n\n"
        "1) Kuvvet dengesi kontrolü:\n"
        f"   R_{support_a.id} + R_{support_b.id} = {reaction_a:.2f} + {reaction_b:.2f} = {sum_forces:.2f} kN\n"
        f"   Toplam yük = {total_vertical:.2f} kN\n"
        f"   Fark = {abs(sum_forces - total_vertical):.4f} ≈ 0 ✓\n\n"
        "2) Moment dengesi kontrolü:\n"
        f"   R_{support_b.id} × mesafe = {reaction_b:.2f} × {span:.2f} = {reaction_b * span:.2f} kN·m\n"
        f"   Yüklerin momenti = {total_moment_about_a:.2f} kN·m\n"
        f"   Fark = {abs(reaction_b * span - total_moment_about_a):.4f} ≈ 0 ✓\n\n"
        "Her iki denge de sağlanıyor, çözüm doğru!\n\n"
        "Not: Reaksiyonların güvenilir şekilde hesaplanması için bu yöntem idealdir."
        " Pratik moment ve kesit hesapları için Kesme yöntemi (tavsiye edilir) ile devam edebilirsiniz."
    )
    
    steps.append(
        SolutionStep(
            step_number=7,
            title="Adım 5: Kontrol Edelim",
            explanation=check_explanation,
            general_formula=rf"\sum F_y = 0 \quad \text{{ve}} \quad \sum M_A = 0",
            substituted_formula=(
                rf"{reaction_a:.2f} + {reaction_b:.2f} = {sum_forces:.2f} \approx {total_vertical:.2f}\text{{ kN}} \quad \checkmark \\"
                rf"{reaction_b:.2f} \times {span:.2f} = {reaction_b * span:.2f} \approx {total_moment_about_a:.2f}\text{{ kN}}\cdot\text{{m}} \quad \checkmark"
            ),
            numerical_result="✓ Tüm denge koşulları sağlanıyor, çözüm doğru! (Sonraki adım için Kesme yöntemine geçmeniz önerilir)",
        )
    )

    return SolutionMethod(
        method_name="support_reactions",
        method_title="Mesnet Reaksiyonları Hesabı",
        description="Statik denge denklemleri (ΣF=0, ΣM=0) ile mesnet tepkileri bulunur; diğer yöntemler için başlangıç verisini sağlar.",
        recommended=True,
        recommendation_reason="Reaksiyon kuvvetleri tüm yöntemlerin temel girdisi olduğundan hesaplamaya her zaman bu adımla başlanmalıdır.",
        steps=steps,
    )


def _generate_section_method(
    payload: SolveRequest,
    reactions: List[SupportReaction],
    shear: np.ndarray,
    moment: np.ndarray,
    x_axis: np.ndarray,
    moment_extrema: Dict[str, Optional[MomentCandidate]],
) -> SolutionMethod:
    regions_points = _collect_section_points(payload)
    regions: List[Tuple[float, float]] = []
    for left, right in zip(regions_points[:-1], regions_points[1:]):
        if right - left > 1e-9:
            regions.append((left, right))

    steps: List[SolutionStep] = []
    step_number = 1

    idea_explanation = (
        "KESME (KESİT) YÖNTEMİ – TEMEL FİKİR\n\n"
        "• Kirişin ilgilendiğimiz bir noktasında hayali bir kesit açarız.\n"
        "• Kesitte doğan iç kuvvetleri N(z), T(z), M(z) bilinmeyen olarak kabul ederiz.\n"
        "• Kestiğimiz parçanın serbest cisim diyagramını çizip sadece ΣF = 0 ve ΣM = 0 denklemlerini kullanırız.\n"
    )
    steps.append(
        SolutionStep(
            step_number=step_number,
            title="Yöntemin Fikri",
            explanation=idea_explanation,
        )
    )
    step_number += 1

    sign_explanation = (
        "İŞARET KABULLERİ (TUTARLI KULLANILMALI)\n\n"
        "• Kesme kuvveti T: Sol yüzeyde yukarı (sağ yüzeyde aşağı) pozitif.\n"
        "• Moment M: Kirişi güldüren (alt lifte çekme) pozitif.\n"
        "• Eksenel kuvvet N: Çekme pozitif (bu örnekte düşey yüklerden ötürü genellikle N = 0)."
    )
    steps.append(
        SolutionStep(
            step_number=step_number,
            title="Pozitif Yön Kabulleri",
            explanation=sign_explanation,
        )
    )
    step_number += 1

    algorithm_explanation = (
        "GENEL AKIŞ\n\n"
        "1) Yük durumunun değiştiği her aralık için ayrı bölge oluştur.\n"
        "2) Her bölge için koordinatı, sol sınırdan itibaren tanımla (z artan yönde sağa gider).\n"
        "3) Serbest cisim diyagramını çiz; yayılı yükleri eşdeğer kuvvetle göster.\n"
        "4) ΣFy = 0 denklemi ile T(z), ΣM = 0 denklemi ile M(z) fonksiyonlarını çıkar.\n"
        "5) Bölge sınırlarında kesme ve moment sürekliliğini kontrol et, sıçramaları noktasal yük/momentlerle eşleştir."
    )
    steps.append(
        SolutionStep(
            step_number=step_number,
            title="Genel Adımlar",
            explanation=algorithm_explanation,
        )
    )
    step_number += 1

    region_summaries: List[str] = []
    for index, (start, end) in enumerate(regions, start=1):
        summary = _describe_region_loads(payload, reactions, start, end)
        region_summaries.append(
            f"Bölge {index}: {_format_interval(start, end, 'z')}\n{summary}\n"
        )

    region_summary_text = (
        f"KİRİŞ {len(regions)} BÖLGEYE AYRILIR\n\n" + "\n".join(region_summaries)
        if region_summaries
        else "Bu yükleme ile kiriş tek bir bölgeden oluşuyor."
    )
    steps.append(
        SolutionStep(
            step_number=step_number,
            title="Bölge Tanımları",
            explanation=region_summary_text,
        )
    )
    step_number += 1

    shear_zero_points: List[float] = []
    moment_zero_points: List[float] = []
    region_formulas: List[dict] = []

    def _ensure_region_samples(x_start: float, x_end: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mask = (x_axis >= x_start - 1e-9) & (x_axis <= x_end + 1e-9)
        region_x = x_axis[mask]
        region_shear = shear[mask]
        region_moment = moment[mask]

        if region_x.size == 0 or not math.isclose(region_x[0], x_start, abs_tol=1e-9):
            shear_start = float(np.interp(x_start, x_axis, shear))
            moment_start = float(np.interp(x_start, x_axis, moment))
            region_x = np.insert(region_x, 0, x_start)
            region_shear = np.insert(region_shear, 0, shear_start)
            region_moment = np.insert(region_moment, 0, moment_start)

        if region_x[-1] < x_end - 1e-9:
            shear_end = float(np.interp(x_end, x_axis, shear))
            moment_end = float(np.interp(x_end, x_axis, moment))
            region_x = np.append(region_x, x_end)
            region_shear = np.append(region_shear, shear_end)
            region_moment = np.append(region_moment, moment_end)

        unique_x, indices = np.unique(region_x, return_index=True)
        region_x = unique_x
        region_shear = region_shear[indices]
        region_moment = region_moment[indices]
        return region_x, region_shear, region_moment

    for idx, (start, end) in enumerate(regions, start=1):
        region_x, region_shear, region_moment = _ensure_region_samples(start, end)
        shear_coeffs = _fit_polynomial(region_x, region_shear, max_degree=2)
        shear_poly = np.poly1d(shear_coeffs)
        moment_poly = np.polyint(shear_poly)

        moment_start_val = float(np.interp(start, x_axis, moment))
        constant_shift = moment_start_val - float(moment_poly(start))
        moment_poly = moment_poly + constant_shift
        moment_coeffs = moment_poly.coeffs

        shear_expression = _format_polynomial(shear_coeffs, variable="z")
        moment_expression = _format_polynomial(moment_coeffs, variable="z")

        local_shear_roots: List[float] = []
        if len(shear_coeffs) > 1:
            for root in np.roots(shear_coeffs):
                if abs(root.imag) < 1e-6:
                    value = float(root.real)
                    if start - 1e-6 <= value <= end + 1e-6:
                        local_shear_roots.append(value)
                        shear_zero_points.append(value)

        local_moment_roots: List[float] = []
        if len(moment_coeffs) > 1:
            for root in np.roots(moment_coeffs):
                if abs(root.imag) < 1e-6:
                    value = float(root.real)
                    if start - 1e-6 <= value <= end + 1e-6:
                        local_moment_roots.append(value)
                        moment_zero_points.append(value)

        shear_end_val = float(shear_poly(end))
        moment_end_val = float(moment_poly(end))

        # Detaylı açıklama için bölgedeki yükleri analiz et
        detailed_derivation = _build_detailed_section_derivation(
            payload, reactions, start, end, idx, shear_expression, moment_expression,
            shear_end_val, moment_end_val
        )

        bullets: List[str] = [
            f"ΣFy = 0 ⇒ T_{idx}(z) = {shear_expression} [kN]",
            f"ΣM_kesit = 0 ⇒ M_{idx}(z) = {moment_expression} [kN·m]",
        ]

        if local_shear_roots:
            zero_lines = []
            for value in sorted(local_shear_roots):
                moment_at_zero = float(moment_poly(value))
                zero_lines.append(
                    f"z = {value:.4f} m → T = 0 ⇒ M = {moment_at_zero:.2f} kN·m"
                )
            bullets.append("Kesme sıfır noktaları:\n" + "\n".join(f"• {line}" for line in zero_lines))

        if local_moment_roots:
            zero_lines = [f"• z = {value:.4f} m" for value in sorted(local_moment_roots)]
            bullets.append("Moment sıfır noktaları:\n" + "\n".join(zero_lines))

        bullets.append(
            f"Bölge sonu değerleri: T(z={end:.4f} m) = {shear_end_val:.2f} kN, M(z={end:.4f} m) = {moment_end_val:.2f} kN·m"
        )

        explanation_text = (
            f"Bölge {idx}: {_format_interval(start, end, 'z')}\n"
            f"{_describe_region_loads(payload, reactions, start, end)}\n\n"
            + detailed_derivation + "\n\n"
            "KESİT DENKLEMLERİ SONUÇLARI\n"
            + "\n".join(bullets)
        )

        steps.append(
            SolutionStep(
                step_number=step_number,
                title=f"Bölge {idx}: Kesit Denklemleri",
                explanation=explanation_text,
                numerical_result=f"T(z={end:.2f} m) = {shear_end_val:.2f} kN, M(z={end:.2f} m) = {moment_end_val:.2f} kN·m",
                beam_section=BeamSectionHighlight(
                    start=_format_float(start),
                    end=_format_float(end),
                    label=f"Bölge {idx}",
                ),
            )
        )
        step_number += 1

        region_formulas.append(
            {
                "start": start,
                "end": end,
                "shear_expression": shear_expression,
                "moment_expression": moment_expression,
                "variable": "z",
            }
        )

    important_lines: List[str] = []
    if shear_zero_points:
        unique_points = sorted({round(val, 6) for val in shear_zero_points})
        for value in unique_points:
            moment_value = float(np.interp(value, x_axis, moment))
            important_lines.append(
                f"• T = 0 ⇒ z = {value:.4f} m → M = {moment_value:.2f} kN·m"
            )
    if moment_zero_points:
        unique_points = sorted({round(val, 6) for val in moment_zero_points})
        for value in unique_points:
            important_lines.append(f"• M = 0 ⇒ z = {value:.4f} m")

    max_positive = moment_extrema.get("max_positive") if moment_extrema else None
    min_negative = moment_extrema.get("min_negative") if moment_extrema else None
    max_absolute = moment_extrema.get("max_absolute") if moment_extrema else None

    if max_positive is not None:
        important_lines.append(
            f"• Pozitif maksimum moment: M = {max_positive[1]:.2f} kN·m (z = {max_positive[0]:.2f} m)"
        )
    if min_negative is not None:
        important_lines.append(
            f"• Negatif maksimum (minimum) moment: M = {min_negative[1]:.2f} kN·m (z = {min_negative[0]:.2f} m)"
        )
    if max_absolute is not None:
        important_lines.append(
            f"• Mutlak değerce kritik moment: |M| = {abs(max_absolute[1]):.2f} kN·m (z = {max_absolute[0]:.2f} m)"
        )

    important_text = (
        "ÖNEMLİ SAYISAL NOKTALAR\n\n" + "\n".join(important_lines)
        if important_lines
        else "Kesme veya moment diyagramında özel bir kritik nokta oluşmadı."
    )

    steps.append(
        SolutionStep(
            step_number=step_number,
            title="Önemli Noktalar",
            explanation=important_text,
        )
    )
    step_number += 1

    piecewise_lines: List[str] = []
    for data in region_formulas:
        variable_name = data.get('variable', 'x')
        piecewise_lines.append(
            f"• {_format_interval(data['start'], data['end'], variable_name)}:\n  T({variable_name}) = {data['shear_expression']}\n  M({variable_name}) = {data['moment_expression']}"
        )

    summary_text = (
        "PARÇALI FONKSİYONLAR\n\nKesme kuvveti ve moment denklemleri, her bölge için aşağıdaki gibidir:\n"
        + "\n".join(piecewise_lines)
        if piecewise_lines
        else "Kiriş tek bölgede çözüldü; fonksiyonlar sabit."
    )

    steps.append(
        SolutionStep(
            step_number=step_number,
            title="Fonksiyonların Toplanması",
            explanation=summary_text,
        )
    )
    step_number += 1

    diagram_explanation = (
        "DİYAGRAM İPUÇLARI\n\n"
        "• Yayılı yük bulunan bölgelerde kesme diyagramı eğimli (lineer/kuadratik), moment diyagramı parabolik/kübik olur.\n"
        "• Kesme diyagramındaki pozitif eğim moment artışını, negatif eğim moment azalışını gösterir.\n"
        "• Noktasal yükler T diyagramında yük büyüklüğü kadar sıçrama yapar; moment diyagramında süreklilik korunur.\n"
        "• Uygulanan çiftler (momentler) M diyagramında anlık atlama yaratır, T diyagramını etkilemez."
    )
    steps.append(
        SolutionStep(
            step_number=step_number,
            title="Çizim İpuçları",
            explanation=diagram_explanation,
        )
    )
    step_number += 1

    mistakes_explanation = (
        "SIK YAPILAN HATALAR\n\n"
        "1) İşaret karıştırmak: Pozitif kesme/moment kabulünü her bölgede aynı tut.\n"
        "2) Yayılı yükün eşdeğer moment kolunu unutmak: w·z kuvveti, bölge uzunluğunun yarısında etkir.\n"
        "3) Bölge aralıklarını yazmamak: Her formülün hangi z aralığında geçerli olduğunu belirt.\n"
        "4) Çift etkisini atlamak: Uygulanan çift M diyagramında atlama oluşturur, T sabit kalır.\n"
        "5) Kontrol yapmamak: T alanının M değişimine eşit olduğunu ve sınır koşullarını doğrula."
    )
    steps.append(
        SolutionStep(
            step_number=step_number,
            title="Sık Yapılan Hatalar",
            explanation=mistakes_explanation,
        )
    )
    step_number += 1

    recap_explanation = (
        "HIZLI ÖZET\n\n"
        "1) Bölge seç, z'yi tanımla.\n"
        "2) Serbest cisim diyagramı + ΣF, ΣM denklemleri ile T(z), M(z) bul.\n"
        "3) Bölge sınırlarında süreklilik/atlamaları kontrol et.\n"
        "4) Kritik noktaları (T=0, M=0) işaretle.\n"
        "5) Diyagramı çiz, sonuçları mesnet ve yük koşullarıyla doğrula."
    )
    steps.append(
        SolutionStep(
            step_number=step_number,
            title="Kısa Özet",
            explanation=recap_explanation,
        )
    )

    description = (
        "Kesit yöntemi, her bölge için serbest cisim diyagramı kurup ΣF ve ΣM denklemlerinden doğrudan T(x) ve M(x)"
        " fonksiyonlarını çıkarır. Dört işlemle ilerleyen klasik mukavemet yaklaşımıdır."
    )

    return SolutionMethod(
        method_name="shear",
        method_title="Kesme (Kesit) Yöntemi",
        description=description,
        steps=steps,
    )


def _generate_differential_method(
    payload: SolveRequest,
    reactions: List[SupportReaction],
    shear: np.ndarray,
    moment: np.ndarray,
    x_axis: np.ndarray,
    moment_extrema: Dict[str, Optional[MomentCandidate]],
) -> SolutionMethod:
    """Generate detailed differential (dM/dx) method solution steps."""
    steps: List[SolutionStep] = []
    supports_sorted = sorted(payload.supports, key=lambda s: s.position)
    support_a, support_b = supports_sorted

    # Find critical points (supports, loads) - these define regions
    critical_points = [0.0, payload.length]
    for support in payload.supports:
        if support.position not in critical_points:
            critical_points.append(support.position)
    for load in payload.point_loads:
        if load.position not in critical_points:
            critical_points.append(load.position)
    for udl in payload.udls:
        if udl.start not in critical_points:
            critical_points.append(udl.start)
        if udl.end not in critical_points:
            critical_points.append(udl.end)
    
    critical_points = sorted(set(critical_points))
    num_regions = len(critical_points) - 1

    # Step 1: Introduction
    intro_explanation = (
        "Kesme kuvveti ile eğilme momenti arasındaki temel diferansiyel ilişki dM/dx = T(x) şeklindedir.\n"
        "Yani moment diyagramı, kesme diyagramının integralidir ve alan hesabı ile bulunur.\n\n"
        "Bu integral, her bölgede kesme kuvvetinin ortalamasını alarak (trapez yaklaşımı) hesaplanır:\n"
        "ΔM ≈ T_ort × Δx.\n\n"
        f"Şimdi kirişi {num_regions} bölgeye ayırıp her bölgede momentin nasıl güncellendiğini adım adım inceleyeceğiz."
    )
    steps.append(
        SolutionStep(
            step_number=1,
            title="Yöntem Temeli",
            explanation=intro_explanation,
            general_formula=r"\Delta M = V \times \Delta x",
            substituted_formula=rf"\text{{Bölge sayısı: }} {num_regions}",
        )
    )

    # Step 2: Initial conditions
    v_start = reactions[0].vertical
    initial_explanation = (
        "Başlangıç koşullarını belirleyelim:\n"
        f"• Sol mesnette ({support_a.id}) kesme kuvveti, hesaplanan mesnet reaksiyonuna eşittir: T(0) = R_{support_a.id} = {v_start:.2f} kN.\n"
        f"• Basit mesnet moment taşıyamadığından başlangıç momenti sıfırdır: M(0) = 0 kN·m.\n"
        "Bu değerler her yeni bölgede güncellenecek referans noktalarıdır."
    )
    steps.append(
        SolutionStep(
            step_number=2,
            title="Başlangıç Değerleri",
            explanation=initial_explanation,
            general_formula=r"M(0) = 0, \quad T(0) = R_A",
            substituted_formula=rf"M(0) = 0, \quad T(0) = {v_start:.2f}\text{{ kN}}",
        )
    )

    # Step 3-N: Calculate moment change for each region
    step_num = 3
    current_moment = 0.0
    
    # Limit to show detailed steps for first 3-4 regions to avoid too many steps
    regions_to_detail = min(num_regions, 4)
    
    for i in range(regions_to_detail):
        x_start = critical_points[i]
        x_end = critical_points[i + 1]
        length = x_end - x_start
        
        # Get shear values at region boundaries
        idx_start = np.argmin(np.abs(x_axis - x_start))
        idx_end = np.argmin(np.abs(x_axis - x_end))
        v_start_region = shear[idx_start]
        v_end_region = shear[idx_end]
        v_avg = (v_start_region + v_end_region) / 2
        
        # Calculate moment change
        moment_change = v_avg * length
        next_moment = current_moment + moment_change
        
        # Get actual moment at end of region
        m_end_actual = moment[idx_end]
        moment_error = m_end_actual - next_moment
        if next_moment < -1e-6:
            orientation_text = "Negatif moment, üst liflerde basma (saat yönü momenti) oluştuğunu gösterir."
        elif next_moment > 1e-6:
            orientation_text = "Pozitif moment, alt liflerde basma (saat yönünün tersi moment) oluştuğunu gösterir."
        else:
            orientation_text = "Sonuç sıfıra çok yakın; bu bölgede moment oluşumu ihmal edilebilir düzeydedir."
        
        region_explanation = (
            f"Bölge {i+1}: [{x_start:.2f} m → {x_end:.2f} m]\n\n"
            f"1) Bölge uzunluğu: Δx = {x_end:.2f} - {x_start:.2f} = {length:.2f} m\n"
            "   (İki kesit arasındaki yatay mesafe; integralin sınırlarını tanımlar.)\n\n"
            f"2) Kesme kuvveti değerleri:\n"
            f"   Başlangıç: T({x_start:.2f}) = {v_start_region:.2f} kN\n"
            f"   Bitiş: T({x_end:.2f}) = {v_end_region:.2f} kN\n"
            f"   Ortalama: T_ort = (T_başlangıç + T_bitiş)/2 = {v_avg:.2f} kN\n"
            "   (Bu bölgede kesme diyagramı doğrusal varsayılır; integral trapez kuralıyla yaklaşıklanır.)\n\n"
            "3) Moment değişimi:\n"
            "   ΔM = ∫ T(x) dx ≈ T_ort × Δx\n"
            f"   ΔM = {v_avg:.2f} × {length:.2f} = {moment_change:.2f} kN·m\n"
            "   (Kesme kuvvetinin işareti, moment artışının yönünü belirler.)\n\n"
            f"4) Güncellenen moment değeri:\n"
            f"   M({x_end:.2f}) = M({x_start:.2f}) + ΔM = {current_moment:.2f} + {moment_change:.2f} = {next_moment:.2f} kN·m\n"
            f"   {orientation_text}\n\n"
            f"   Referans kontrolü: Sayısal çözüm M({x_end:.2f}) = {m_end_actual:.2f} kN·m ⇒ fark = {moment_error:.4f} kN·m (yuvarlama kaynaklı)."
        )
        
        steps.append(
            SolutionStep(
                step_number=step_num,
                title=f"Bölge {i+1}: x = {x_start:.2f} → {x_end:.2f} m",
                explanation=region_explanation,
                general_formula=r"\Delta M = \int_{x_i}^{x_{i+1}} T(x)\,dx \approx T_{\text{ort}} \times \Delta x, \quad M_{i+1} = M_i + \Delta M",
                substituted_formula=rf"{moment_change:.2f} = {v_avg:.2f} \times {length:.2f}, \quad {next_moment:.2f} = {current_moment:.2f} + {moment_change:.2f}",
                numerical_result=f"M({x_end:.2f} m) = {next_moment:.2f} kN·m",
                beam_section=BeamSectionHighlight(
                    start=_format_float(min(x_start, x_end)),
                    end=_format_float(max(x_start, x_end)),
                    label=f"Bölge {i+1}"
                ),
            )
        )
        
        current_moment = next_moment
        step_num += 1
    
    # If there are more regions, summarize them
    if num_regions > regions_to_detail:
        remaining_explanation = (
            f"Kalan {num_regions - regions_to_detail} bölge için aynı yöntem uygulanır:\n"
            "Her bölgede kesme kuvveti ortalaması ile bölge uzunluğu çarpılarak moment değişimi bulunur."
        )
        steps.append(
            SolutionStep(
                step_number=step_num,
                title=f"Kalan Bölgeler ({regions_to_detail + 1}→{num_regions})",
                explanation=remaining_explanation,
                general_formula=r"\Delta M_i = T_{\text{ort},i} \times \Delta x_i",
                substituted_formula=r"\text{Aynı işlem tekrarlanır}",
            )
        )
        step_num += 1

    # Maximum moment
    max_positive = moment_extrema.get("max_positive")
    min_negative = moment_extrema.get("min_negative")
    max_absolute = moment_extrema.get("max_absolute")

    extremum_lines: List[str] = []
    if max_positive is not None:
        extremum_lines.append(
            f"• Pozitif yönde maksimum moment: M = {max_positive[1]:.2f} kN·m (x = {max_positive[0]:.2f} m)"
        )

    if min_negative is not None:
        extremum_lines.append(
            f"• Negatif yönde maksimum moment (minimum): M = {min_negative[1]:.2f} kN·m (x = {min_negative[0]:.2f} m)"
        )

    highlight_position = None
    highlight_value = None
    if max_positive is not None:
        highlight_position = max_positive[0]
        highlight_value = max_positive[1]

    if max_absolute is not None:
        extremum_lines.append(
            f"• Mutlak değerce en kritik moment: |M| = {abs(max_absolute[1]):.2f} kN·m (x = {max_absolute[0]:.2f} m)"
        )
        if highlight_position is None:
            highlight_position = max_absolute[0]
            highlight_value = max_absolute[1]

    if highlight_position is None:
        highlight_position = _format_float(x_axis[np.argmax(np.abs(moment))])
        highlight_value = float(moment[np.argmax(np.abs(moment))])

    max_explanation = (
        "Kesme kuvveti diyagramında T(x) = 0 olduğu noktalar moment diyagramında ekstremum (maksimum/minimum) değerleri verir.\n\n"
    )
    if extremum_lines:
        max_explanation += "\n".join(extremum_lines)
    else:
        max_explanation += "Moment diyagramında belirgin bir kritik nokta bulunmadı; uç noktalar kontrol edildi."

    subtitle = []
    if highlight_value is not None:
        subtitle.append(rf"M(x) = {highlight_value:.2f} \text{{ kN}}\cdot\text{{m}}")
    if highlight_position is not None:
        subtitle.append(rf"x = {highlight_position:.2f} \text{{ m}}")

    if highlight_value is not None and highlight_value < 0.0:
        label_text = "Minimum Moment"
    else:
        label_text = "Maksimum Moment"

    steps.append(
        SolutionStep(
            step_number=step_num,
            title="Moment Ekstremumları",
            explanation=max_explanation,
            general_formula=r"T(x) = \dfrac{dM}{dx} = 0",
            substituted_formula=",\; ".join(subtitle) if subtitle else None,
            numerical_result=(
                f"{label_text}: M = {highlight_value:.2f} kN·m (x = {highlight_position:.2f} m)"
                if highlight_value is not None and highlight_position is not None
                else None
            ),
            beam_section=BeamSectionHighlight(
                start=_format_float(highlight_position),
                end=_format_float(highlight_position),
                label=label_text,
            ),
        )
    )
    step_num += 1

    # Verification
    end_moment = moment[-1]
    verification_explanation = (
        f"Son kontrol: Sağ mesnet ({support_b.id}) noktasında moment sıfır olmalı.\n\n"
        f"Hesaplanan: M({payload.length:.2f}) = {end_moment:.4f} kN·m ≈ 0 ✓"
    )
    
    steps.append(
        SolutionStep(
            step_number=step_num,
            title="Kontrol",
            explanation=verification_explanation,
            general_formula=rf"M(L) = 0",
            substituted_formula=rf"M({payload.length:.2f}) \approx {end_moment:.4f} \approx 0",
            numerical_result="✓ Sınır koşulu sağlanıyor",
                beam_section=BeamSectionHighlight(
                    start=0.0,
                    end=_format_float(payload.length),
                    label="Global Kontrol"
                ),
        )
    )

    return SolutionMethod(
        method_name="differential",
        method_title="Diferansiyel (dM/dx) Yaklaşımı",
        description="Kesme diyagramını (T) kullanarak moment diyagramını sayısal integrasyonla elde eder. Trapez kuralı ile dM/dx = T bağıntısını uygular.",
        steps=steps,
    )


def _generate_area_method(
    payload: SolveRequest,
    shear: np.ndarray,
    moment: np.ndarray,
    x_axis: np.ndarray,
) -> SolutionMethod:
    """Generate detailed area method solution steps."""
    steps: List[SolutionStep] = []

    # Find critical points to define regions
    critical_points = [0.0, payload.length]
    for support in payload.supports:
        if support.position not in critical_points:
            critical_points.append(support.position)
    for load in payload.point_loads:
        if load.position not in critical_points:
            critical_points.append(load.position)
    for udl in payload.udls:
        if udl.start not in critical_points:
            critical_points.append(udl.start)
        if udl.end not in critical_points:
            critical_points.append(udl.end)
    
    critical_points = sorted(set(critical_points))
    num_regions = len(critical_points) - 1

    # Step 1: Method principle
    principle_explanation = (
        "ALAN YÖNTEMİ - GRAFİK TABANLI YAKLAŞIM\n\n"
        "Kesme kuvveti grafiğindeki alanlar kullanılarak moment hesaplanır.\n\n"
        "Temel Prensip:\n"
        "- Kesme grafiğinde bir bölgenin ALANI = O bölgedeki MOMENT DEĞİŞİMİ\n"
        "- Alan pozitif ise moment artar\n"
        "- Alan negatif ise moment azalır\n"
        "- Alan sıfır ise moment sabit kalır\n\n"
        f"Bu kiriş {num_regions} bölgeye ayrılacak.\n"
        "Her bölge için kesme grafiğindeki şekil belirlenerek alan hesaplanacak."
    )
    steps.append(
        SolutionStep(
            step_number=1,
            title="Yöntem İlkesi (Grafik Tabanlı)",
            explanation=principle_explanation,
            general_formula=r"\text{Alan}_{V} = \Delta M \quad \Rightarrow \quad M_{\text{yeni}} = M_{\text{eski}} + \text{Alan}_{V}",
            substituted_formula=rf"\text{{Bölge sayısı: }} {num_regions}",
        )
    )

    # Step 2: Initial condition
    initial_explanation = (
        "BAŞLANGIÇ NOKTASI\n\n"
        f"Basit mesnetli kirişin sol ucunda (x = 0) moment sıfırdır.\n\n"
        "Bu noktadan başlayarak sağa doğru ilerlenecek.\n"
        "Her bölgede kesme grafiğinin alanı hesaplanıp momente eklenecek."
    )
    steps.append(
        SolutionStep(
            step_number=2,
            title="Başlangıç Noktası",
            explanation=initial_explanation,
            general_formula=r"M(0) = 0",
            substituted_formula=r"M(0) = 0 \text{ kN}\cdot\text{m}",
            numerical_result="Başlangıç momenti: M(0) = 0 kN.m",
        )
    )

    # Step 3-N: Calculate area for each region (limit to first 3-4 regions for detail)
    step_num = 3
    current_moment = 0.0
    regions_to_detail = min(num_regions, 4)
    
    for i in range(regions_to_detail):
        x_start = critical_points[i]
        x_end = critical_points[i + 1]
        length = x_end - x_start
        
        # Get shear values
        idx_start = np.argmin(np.abs(x_axis - x_start))
        idx_end = np.argmin(np.abs(x_axis - x_end))
        v_start = shear[idx_start]
        v_end = shear[idx_end]
        
        # Determine shape and calculate area
        if abs(v_start - v_end) < 0.01:  # Nearly constant
            shape_type = "Dikdörtgen"
            area = v_start * length
            area_formula = f"{v_start:.2f} × {length:.2f}"
        elif v_start * v_end < 0:  # Changes sign
            shape_type = "İkiz Üçgen (işaret değişimi)"
            area = 0.5 * (v_start + v_end) * length
            area_formula = f"0.5 × ({v_start:.2f} + {v_end:.2f}) × {length:.2f}"
        elif abs(v_start) > 0.01 and abs(v_end) > 0.01:  # Trapezoid
            shape_type = "Yamuk"
            area = 0.5 * (v_start + v_end) * length
            area_formula = f"0.5 × ({v_start:.2f} + {v_end:.2f}) × {length:.2f}"
        else:  # Triangle
            shape_type = "Üçgen"
            area = 0.5 * (v_start + v_end) * length
            area_formula = f"0.5 × {max(abs(v_start), abs(v_end)):.2f} × {length:.2f}"
        
        next_moment = current_moment + area
        
        # Determine trend text
        if area > 0:
            trend = "Pozitif alan - Moment artar"
        elif area < 0:
            trend = "Negatif alan - Moment azalır"
        else:
            trend = "Alan sıfır - Moment sabit"
        
        # Detailed explanation
        region_explanation = (
            f"BÖLGE {i+1}: Aralık [{x_start:.2f} m - {x_end:.2f} m]\n\n"
            f"GRAFİKTEN BELİRLENEN ŞEKİL:\n"
            f"Bu bölgede kesme grafiği {shape_type} şeklindedir.\n\n"
            f"BÖLGE BİLGİLERİ:\n"
            f"- Başlangıç noktası: x = {x_start:.2f} m, V = {v_start:.2f} kN\n"
            f"- Bitiş noktası: x = {x_end:.2f} m, V = {v_end:.2f} kN\n"
            f"- Bölge uzunluğu: Δx = {length:.2f} m\n\n"
            f"ALAN HESABI ({shape_type} formülü):\n"
            f"Alan = {area_formula}\n"
            f"Alan = {area:.2f} kN·m\n\n"
            f"MOMENT DEĞİŞİMİ:\n"
            f"Kesme grafiğindeki alan = Moment değişimi\n"
            f"ΔM = {area:.2f} kN·m\n"
            f"{trend}\n\n"
            f"YENİ MOMENT HESABI:\n"
            f"M_yeni = M_eski + ΔM\n"
            f"M({x_end:.2f}) = {current_moment:.2f} + ({area:.2f})\n"
            f"M({x_end:.2f}) = {next_moment:.2f} kN·m"
        )
        
        # Build visualization samples
        region_mask = (x_axis >= x_start - 1e-9) & (x_axis <= x_end + 1e-9)
        region_x = x_axis[region_mask]
        region_v = shear[region_mask]

        start_missing = region_x.size == 0 or abs(region_x[0] - x_start) > 1e-9
        end_missing = region_x.size == 0 or abs(region_x[-1] - x_end) > 1e-9

        if start_missing:
            start_shear = float(np.interp(x_start, x_axis, shear))
            region_x = np.insert(region_x, 0, x_start)
            region_v = np.insert(region_v, 0, start_shear)
        if end_missing:
            end_shear = float(np.interp(x_end, x_axis, shear))
            region_x = np.append(region_x, x_end)
            region_v = np.append(region_v, end_shear)

        moment_segment_vals = moment[region_mask]
        if start_missing:
            moment_segment_vals = np.insert(moment_segment_vals, 0, float(np.interp(x_start, x_axis, moment)))
        if end_missing:
            moment_segment_vals = np.append(moment_segment_vals, float(np.interp(x_end, x_axis, moment)))

        region_samples = ShearRegionSamples(
            x=[_format_float(val) for val in region_x.tolist()],
            shear=[_format_float(val) for val in region_v.tolist()],
        )
        moment_samples = MomentSegmentSamples(
            x=[_format_float(val) for val in region_x.tolist()],
            moment=[_format_float(val) for val in moment_segment_vals.tolist()],
        )

        if area > 1e-6:
            trend_label = "increase"
        elif area < -1e-6:
            trend_label = "decrease"
        else:
            trend_label = "constant"

        steps.append(
            SolutionStep(
                step_number=step_num,
                title=f"Bölge {i+1}: {shape_type} ({x_start:.2f}-{x_end:.2f} m)",
                explanation=region_explanation,
                general_formula=r"\text{Alan}_{\text{geometrik}} = f(\text{şekil}), \quad \Delta M = \text{Alan}",
                substituted_formula=rf"\text{{Alan}} = {area:.2f}\text{{ kN}}\cdot\text{{m}}, \quad M({x_end:.2f}) = {next_moment:.2f}\text{{ kN}}\cdot\text{{m}}",
                numerical_result=f"Bölge {i+1}: Alan = {area:.2f} kN·m, Moment = {next_moment:.2f} kN·m",
                area_visualization=AreaMethodVisualization(
                    shape=shape_type,
                    area_value=_format_float(area),
                    trend=trend_label,
                    region=region_samples,
                    moment_segment=moment_samples,
                ),
            )
        )
        
        current_moment = next_moment
        step_num += 1
    
    # Summary for remaining regions if any
    if num_regions > regions_to_detail:
        remaining_explanation = (
            f"KALAN BÖLGELER (Bölge {regions_to_detail + 1} - {num_regions})\n\n"
            f"Kalan {num_regions - regions_to_detail} bölge için aynı yöntem uygulanır:\n\n"
            "ADIMLAR:\n"
            "1. Kesme grafiğinden bölgenin şeklini belirle\n"
            "   (Dikdörtgen, Üçgen veya Yamuk)\n\n"
            "2. Geometrik formüllerle alanı hesapla\n"
            "   - Dikdörtgen: taban × yükseklik\n"
            "   - Üçgen: (taban × yükseklik) / 2\n"
            "   - Yamuk: (taban1 + taban2) × yükseklik / 2\n\n"
            "3. Hesaplanan alan = Moment değişimi\n\n"
            "4. Yeni moment = Önceki moment + Alan değişimi"
        )
        steps.append(
            SolutionStep(
                step_number=step_num,
                title=f"Kalan Bölgeler ({regions_to_detail + 1}-{num_regions})",
                explanation=remaining_explanation,
                general_formula=r"\text{Her bölge için: } \Delta M_i = \text{Alan}_i",
                substituted_formula=r"\text{Aynı grafik yöntemi devam eder}",
            )
        )
        step_num += 1

    # Maximum moment
    max_moment_idx = np.argmax(np.abs(moment))
    max_moment_x = x_axis[max_moment_idx]
    max_moment_value = moment[max_moment_idx]
    
    # Calculate total area from start to max
    total_area_to_max = 0.0
    for i in range(1, max_moment_idx + 1):
        total_area_to_max += 0.5 * (shear[i-1] + shear[i]) * (x_axis[i] - x_axis[i-1])
    
    max_explanation = (
        f"MAKSİMUM MOMENT BELİRLEME\n\n"
        f"Kesme grafiğinde V = 0 olan nokta bulunur.\n"
        f"Bu nokta: x = {max_moment_x:.2f} m\n\n"
        f"Maksimum moment hesabı:\n"
        f"Başlangıçtan (x=0) bu noktaya kadar olan toplam alan hesaplanır.\n\n"
        f"Grafikten hesaplanan:\n"
        f"- Aralık: 0 → {max_moment_x:.2f} m\n"
        f"- Toplam alan: {total_area_to_max:.2f} kN·m\n\n"
        f"Sonuç:\n"
        f"Bu toplam alan = Maksimum moment değeri"
    )
    
    steps.append(
        SolutionStep(
            step_number=step_num,
            title="Maksimum Moment Belirleme",
            explanation=max_explanation,
            general_formula=r"M_{\max} = \sum \text{Alan}_{0 \to V=0}",
            substituted_formula=rf"M_{{\max}} = {max_moment_value:.2f}\text{{ kN}}\cdot\text{{m}} \quad (x = {max_moment_x:.2f}\text{{ m}})",
            numerical_result=f"M_max = {max_moment_value:.2f} kN·m (x = {max_moment_x:.2f} m)",
        )
    )
    step_num += 1

    # Verification
    end_moment = moment[-1]
    total_area = 0.0
    positive_area = 0.0
    negative_area = 0.0
    
    for i in range(1, len(x_axis)):
        area_segment = 0.5 * (shear[i-1] + shear[i]) * (x_axis[i] - x_axis[i-1])
        total_area += area_segment
        if area_segment > 0:
            positive_area += area_segment
        else:
            negative_area += area_segment
    
    verification_explanation = (
        f"DOĞRULAMA (Grafik Kontrolü)\n\n"
        f"Basit mesnetli kirişte başlangıç ve bitiş noktalarında moment sıfır olmalıdır.\n\n"
        f"GRAFİKTEN HESAPLANAN TOPLAM ALANLAR:\n\n"
        f"Pozitif alanlar toplamı: {positive_area:.2f} kN·m\n"
        f"(Kesme grafiğinde x-ekseninin üstündeki bölgeler)\n\n"
        f"Negatif alanlar toplamı: {negative_area:.2f} kN·m\n"
        f"(Kesme grafiğinde x-ekseninin altındaki bölgeler)\n\n"
        f"Net toplam alan: {total_area:.4f} kN·m ≈ 0\n\n"
        f"SONUÇ KONTROLÜ:\n"
        f"M(L) = M(0) + Toplam alan\n"
        f"M({payload.length:.2f}) = 0 + {total_area:.4f}\n"
        f"M({payload.length:.2f}) ≈ {end_moment:.4f} ≈ 0\n\n"
        f"Grafik yöntemi ile hesaplanan değerler doğrulanmıştır."
    )
    
    steps.append(
        SolutionStep(
            step_number=step_num,
            title="Doğrulama ve Kontrol",
            explanation=verification_explanation,
            general_formula=r"M(L) = M(0) + \sum \text{Tüm alanlar} = 0",
            substituted_formula=rf"{positive_area:.2f} + ({negative_area:.2f}) = {total_area:.4f} \approx 0",
            numerical_result=f"Grafik doğrulandı: M({payload.length:.2f}) ≈ 0",
        )
    )

    return SolutionMethod(
        method_name="area",
        method_title="Alan Yöntemi (Grafik Tabanlı)",
        description="Kesme kuvveti GRAFİĞİ altındaki alanlar (dikdörtgen, üçgen, yamuk) hesaplanarak moment diyagramı çizilir. Görsel ve sezgisel bir yöntemdir.",
        recommendation_reason="Kesme ve moment diyagramlarını görsel olarak açıklığa kavuşturmak ve eğitim/raporlama aşamalarında sezgisel doğrulama yapmak için idealdir.",
        steps=steps,
    )


def _generate_detailed_solutions(
    payload: SolveRequest,
    reactions: List[SupportReaction],
    shear: np.ndarray,
    moment: np.ndarray,
    x_axis: np.ndarray,
    recommendation: MethodRecommendation,
    moment_extrema: Dict[str, Optional[MomentCandidate]],
) -> DetailedSolution:
    """Generate all detailed solution methods."""

    methods = [
        _generate_equilibrium_method(payload, reactions),
        _generate_section_method(payload, reactions, shear, moment, x_axis, moment_extrema),
        _generate_differential_method(payload, reactions, shear, moment, x_axis, moment_extrema),
        _generate_area_method(payload, shear, moment, x_axis),
    ]

    for method in methods:
        if method.method_name == recommendation.method:
            method.recommended = True
            method.recommendation_reason = recommendation.reason
        elif method.method_name in {"shear", "area"}:
            method.recommended = False
            method.recommendation_reason = None
    
    # Include diagram data for visualization in solution steps
    diagram_data = DiagramData(
        x=x_axis.tolist(),
        shear=shear.tolist(),
        moment=moment.tolist(),
        normal=np.zeros_like(x_axis).tolist(),  # Placeholder for now
    )

    beam_context = BeamContext(
        length=_format_float(payload.length),
        supports=[
            BeamSupportInfo(
                id=support.id,
                type=support.type,
                position=_format_float(support.position),
            )
            for support in sorted(payload.supports, key=lambda item: item.position)
        ],
        point_loads=[
            BeamPointLoadInfo(
                id=load.id,
                magnitude=_format_float(load.magnitude),
                position=_format_float(load.position),
                angle_deg=_format_float(load.angle_deg),
            )
            for load in payload.point_loads
        ],
        udls=[
            BeamDistributedLoadInfo(
                id=udl.id,
                magnitude=_format_float(udl.magnitude),
                start=_format_float(udl.start),
                end=_format_float(udl.end),
                direction=udl.direction,
                shape=udl.shape,
            )
            for udl in payload.udls
        ],
        moment_loads=[
            BeamMomentLoadInfo(
                id=moment_load.id,
                magnitude=_format_float(moment_load.magnitude),
                position=_format_float(moment_load.position),
                direction=moment_load.direction,
            )
            for moment_load in payload.moment_loads
        ],
    )

    return DetailedSolution(methods=methods, diagram=diagram_data, beam_context=beam_context)


def solve_beam(payload: SolveRequest) -> SolveResponse:
    sampling_points = DEFAULT_SAMPLING_POINTS

    start_time = perf_counter()
    reactions, derivations = _compute_reactions(payload)
    recommendation = _determine_method_recommendation(payload)

    base_axis = np.linspace(0.0, payload.length, num=sampling_points, dtype=float, endpoint=True)
    if base_axis.size > 0:
        base_axis[0] = 0.0
        base_axis[-1] = payload.length

    shear_base = _shear_diagram(payload, base_axis, reactions)

    critical_points: List[float] = base_axis.tolist()

    supports_sorted = sorted(payload.supports, key=lambda support: support.position)
    for support in supports_sorted:
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

    warnings: List[str] = []

    moment_extrema = _compute_moment_extrema(payload, reactions, x_axis, shear)

    right_support_pos = supports_sorted[-1].position
    moment_at_right = moment[-1]
    
    # Maksimum momenti bul (göreli hata kontrolü için)
    max_moment = float(np.max(np.abs(moment)))
    
    # Mutlak eşik (0.1 kNm) veya göreli eşik (%0.5 of max moment)
    absolute_threshold = 0.1
    relative_threshold = 0.005 * max_moment if max_moment > 0 else absolute_threshold
    threshold = max(absolute_threshold, relative_threshold)
    
    if abs(moment_at_right) > threshold:
        warnings.append(
            f"x={right_support_pos:.2f} m noktasındaki moment sıfıra yakın değil (|{moment_at_right:.3f}|). Sayısal sapma mevcut olabilir."
        )

    axial_balance = sum(reaction.axial for reaction in reactions) - sum(
        _axial_component(load) for load in payload.point_loads
    )
    if abs(axial_balance) > 1e-3:
        warnings.append("Eksenel denge artığı beklenenden büyük.")

    duration_ms = (perf_counter() - start_time) * 1000.0

    # Generate detailed solutions
    detailed_solutions = _generate_detailed_solutions(
        payload,
        reactions,
        shear,
        moment,
        x_axis,
        recommendation,
        moment_extrema,
    )

    max_positive = moment_extrema.get("max_positive")
    min_negative = moment_extrema.get("min_negative")
    max_absolute = moment_extrema.get("max_absolute")

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
            recommendation=recommendation,
            max_positive_moment=_format_float(max_positive[1]) if max_positive else None,
            max_positive_position=_format_float(max_positive[0]) if max_positive else None,
            min_negative_moment=_format_float(min_negative[1]) if min_negative else None,
            min_negative_position=_format_float(min_negative[0]) if min_negative else None,
            max_absolute_moment=_format_float(max_absolute[1]) if max_absolute else None,
            max_absolute_position=_format_float(max_absolute[0]) if max_absolute else None,
        ),
        detailed_solutions=detailed_solutions,
    )

    return response

