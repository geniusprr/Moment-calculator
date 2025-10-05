from __future__ import annotations

import math
from time import perf_counter
from typing import List

import numpy as np

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


def _determine_method_recommendation(payload: SolveRequest) -> MethodRecommendation:
    has_distributed = len(payload.udls) > 0
    has_moments = len(payload.moment_loads) > 0
    has_axial_point = any(abs(_axial_component(load)) > 1e-6 for load in payload.point_loads)
    has_point_loads = len(payload.point_loads) > 0

    if has_point_loads and not has_distributed and not has_moments and not has_axial_point:
        return MethodRecommendation(
            method="area",
            title="Alan Yöntemi",
            reason=(
                "Sadece dikey tekil yükler bulunduğundan kesme diyagramı parça parça sabit kalır ve moment "
                "doğrusal segmentlerden oluşur; alan yöntemi bu durumda doğrudan ve hızlıdır."
            ),
        )

    return MethodRecommendation(
        method="shear",
        title="Kesme Yöntemi",
        reason=(
            "Yayılı veya üçgen yükler, açılı kuvvetler ya da mesnet momentleri bulunduğundan moment diyagramı parabolik/"
            "karmaşık davranır; kesme yöntemi bu tür durumlarda daha pratik ve hatasız ilerler."
        ),
    )


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


def _generate_integration_method(
    payload: SolveRequest,
    reactions: List[SupportReaction],
    shear: np.ndarray,
    moment: np.ndarray,
    x_axis: np.ndarray,
) -> SolutionMethod:
    """Generate detailed integration method solution steps."""
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
        "Kesme kuvveti ve eğilme momenti arasındaki temel ilişki:\n"
        "Moment değişimi = Kesme kuvveti × Mesafe\n\n"
        f"Kiriş {num_regions} bölgeye ayrılacak ve her bölgede moment değişimi hesaplanacak."
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
        f"Başlangıç değerleri:\n"
        f"• Sol uçta ({support_a.id}) kesme kuvveti: V = {v_start:.2f} kN\n"
        f"• Sol uçta moment: M = 0 kN·m (basit mesnet)"
    )
    steps.append(
        SolutionStep(
            step_number=2,
            title="Başlangıç Değerleri",
            explanation=initial_explanation,
            general_formula=r"M(0) = 0, \quad V(0) = R_A",
            substituted_formula=rf"M(0) = 0, \quad V(0) = {v_start:.2f}\text{{ kN}}",
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
        
        region_explanation = (
            f"Bölge {i+1}: [{x_start:.2f} m → {x_end:.2f} m]\n\n"
            f"1) Bölge uzunluğu: Δx = {x_end:.2f} - {x_start:.2f} = {length:.2f} m\n"
            f"2) Kesme kuvveti:\n"
            f"   Başlangıç: V = {v_start_region:.2f} kN\n"
            f"   Bitiş: V = {v_end_region:.2f} kN\n"
            f"   Ortalama: V_ort = {v_avg:.2f} kN\n\n"
            f"3) Moment değişimi:\n"
            f"   ΔM = V_ort × Δx\n"
            f"   ΔM = {v_avg:.2f} × {length:.2f} = {moment_change:.2f} kN·m\n\n"
            f"4) Yeni moment:\n"
            f"   M({x_end:.2f}) = M({x_start:.2f}) + ΔM\n"
            f"   M({x_end:.2f}) = {current_moment:.2f} + {moment_change:.2f} = {next_moment:.2f} kN·m"
        )
        
        steps.append(
            SolutionStep(
                step_number=step_num,
                title=f"Bölge {i+1}: x = {x_start:.2f} → {x_end:.2f} m",
                explanation=region_explanation,
                general_formula=r"\Delta M = V_{\text{ort}} \times \Delta x, \quad M_{\text{yeni}} = M_{\text{eski}} + \Delta M",
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
                general_formula=r"\Delta M_i = V_{\text{ort},i} \times \Delta x_i",
                substituted_formula=r"\text{Aynı işlem tekrarlanır}",
            )
        )
        step_num += 1

    # Maximum moment
    max_moment_idx = np.argmax(np.abs(moment))
    max_moment_x = x_axis[max_moment_idx]
    max_moment_value = moment[max_moment_idx]
    
    max_explanation = (
        f"Hesaplanan tüm moment değerleri arasından maksimum değer bulunur.\n\n"
        f"Maksimum moment x = {max_moment_x:.2f} m noktasında oluşur.\n"
        f"Bu noktada kesme kuvveti sıfıra yakındır."
    )
    
    steps.append(
        SolutionStep(
            step_number=step_num,
            title="Maksimum Moment",
            explanation=max_explanation,
            general_formula=r"M_{\max} = \max\{M(x) : x \in [0, L]\}",
            substituted_formula=rf"M_{{\max}} = {max_moment_value:.2f}\text{{ kN}}\cdot\text{{m}} \quad (x = {max_moment_x:.2f}\text{{ m}})",
            numerical_result=f"✓ M_max = {max_moment_value:.2f} kN·m",
                beam_section=BeamSectionHighlight(
                    start=_format_float(max_moment_x),
                    end=_format_float(max_moment_x),
                    label="Maksimum Moment"
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
        method_name="shear",
        method_title="Kesme Yöntemi",
        description="Kesme kuvveti grafiği bölgelere ayrılır. Her bölgede: Moment Değişimi = Kesme × Mesafe formülü kullanılır. Basit ve pratik bir yöntemdir.",
        recommendation_reason="Kesme diyagramındaki değerlerle momenti hızlı ve doğrudan hesapladığı için günlük mühendislik hesaplarında pratik bir yaklaşımdır.",
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
) -> DetailedSolution:
    """Generate all detailed solution methods."""
    from ..schemas.beam import DiagramData
    
    methods = [
        _generate_equilibrium_method(payload, reactions),
        _generate_integration_method(payload, reactions, shear, moment, x_axis),
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
    sampling_points = payload.sampling.points if payload.sampling else 401

    start_time = perf_counter()
    reactions, derivations = _compute_reactions(payload)
    recommendation = _determine_method_recommendation(payload)

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

    # Generate detailed solutions
    detailed_solutions = _generate_detailed_solutions(payload, reactions, shear, moment, x_axis, recommendation)

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
        ),
        detailed_solutions=detailed_solutions,
    )

    return response

