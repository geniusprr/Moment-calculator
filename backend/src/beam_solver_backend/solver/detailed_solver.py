from typing import List, Optional, Tuple
import math
import numpy as np

from beam_solver_backend.schemas.beam import (
    DetailedSolution,
    SolutionMethod,
    SolutionStep,
    SolveRequest,
    SupportReaction,
    DiagramData,
    BeamContext,
    Support,
    PointLoad,
    UniformDistributedLoad,
    MomentLoad,
    BeamSectionHighlight,
    AreaMethodVisualization,
    ShearRegionSamples,
    MomentSegmentSamples,
)

class DetailedSolver:
    def __init__(
        self, 
        payload: SolveRequest, 
        reactions: List[SupportReaction], 
        diagram: DiagramData
    ):
        self.payload = payload
        self.reactions = reactions
        self.diagram = diagram
        self.beam_context = self._build_beam_context()

    def _build_beam_context(self) -> BeamContext:
        return BeamContext(
            length=self.payload.length,
            supports=[
                Support(
                    id=s.id, 
                    type=s.type, 
                    position=s.position
                ) for s in self.payload.supports
            ],
            point_loads=[
                PointLoad(
                    id=l.id, 
                    magnitude=l.magnitude, 
                    position=l.position, 
                    angle_deg=l.angle_deg
                ) for l in self.payload.point_loads
            ],
            udls=[
                UniformDistributedLoad(
                    id=l.id, 
                    magnitude=l.magnitude, 
                    start=l.start, 
                    end=l.end, 
                    direction=l.direction, 
                    shape=l.shape
                ) for l in self.payload.udls
            ],
            moment_loads=[
                MomentLoad(
                    id=l.id, 
                    magnitude=l.magnitude, 
                    position=l.position, 
                    direction=l.direction
                ) for l in self.payload.moment_loads
            ]
        )

    def solve(self) -> DetailedSolution:
        methods = []
        
        # 1. Support Reactions (Always first)
        methods.append(self._solve_reactions())
        
        # 2. Section Method (Kesme Yöntemi)
        methods.append(self._solve_section_method())
        
        # 3. Area Method (Alan Yöntemi)
        methods.append(self._solve_area_method())
        
        # 4. Differential Method (Diferansiyel Yöntem)
        methods.append(self._solve_differential_method())

        return DetailedSolution(
            methods=methods,
            diagram=self.diagram,
            beam_context=self.beam_context
        )

    def _solve_reactions(self) -> SolutionMethod:
        steps = []
        
        # Step 1: Free Body Diagram explanation
        steps.append(SolutionStep(
            step_number=1,
            title="Serbest Cisim Diyagramı",
            explanation="Kiriş üzerindeki tüm yükler ve mesnet tepkileri belirlenir. Bilinmeyen mesnet tepkileri için yön kabulleri yapılır.",
            beam_section=BeamSectionHighlight(start=0, end=self.payload.length, label="Tüm Sistem")
        ))

        # Step 2: Equilibrium Equations
        steps.append(SolutionStep(
            step_number=2,
            title="Denge Denklemleri",
            explanation="Statik denge için üç temel denklem kullanılır:\n1. Toplam Fx = 0 (Yatay Denge)\n2. Toplam Fy = 0 (Düşey Denge)\n3. Toplam M = 0 (Moment Dengesi)",
            general_formula=r"\sum F_x = 0, \quad \sum F_y = 0, \quad \sum M = 0"
        ))

        # Sort supports by position
        sorted_supports = sorted(self.reactions, key=lambda r: r.position)
        
        if not sorted_supports:
            return SolutionMethod(
                method_name="support_reactions",
                method_title="Mesnet Tepkileri",
                description="Denge denklemleri kullanılarak mesnet tepkilerinin bulunması.",
                steps=steps
            )

        # Moment equation about first support (usually A)
        ref_support = sorted_supports[0]
        other_support = sorted_supports[1] if len(sorted_supports) > 1 else None

        # Build moment equation string
        moment_terms = []
        
        # 1. Other support reaction
        if other_support:
            dist = other_support.position - ref_support.position
            moment_terms.append(f"R_{{{other_support.support_id}}} \\cdot {dist:.2f}")

        # 2. Point loads
        for load in self.payload.point_loads:
            dist = load.position - ref_support.position
            angle_rad = math.radians(load.angle_deg)
            vertical_comp = -load.magnitude * math.sin(angle_rad) # Positive is up usually, but here load mag is positive
            # If load is down (angle -90), vertical_comp is positive magnitude.
            # Moment = Force * dist. 
            # Let's stick to standard: CCW positive.
            # Downward force at right of support creates CW moment (negative).
            
            # Let's simplify: Sum M_A = 0 => R_B * L - P * a - q * L * L/2 = 0
            # So R_B * L = P * a + ...
            
            # We will show: R_B * L + (Moment from loads) = 0
            pass

        # Simplified approach for educational display:
        # Show the general equation structure first
        steps.append(SolutionStep(
            step_number=3,
            title=f"{ref_support.support_id} Noktasına Göre Moment",
            explanation=f"Bilinmeyen sayısını azaltmak için moment dengesi {ref_support.support_id} noktasına göre yazılır. Saat yönünün tersi (CCW) pozitif kabul edilir.",
            general_formula=rf"\sum M_{{{ref_support.support_id}}} = 0 \Rightarrow \sum (F \cdot d) = 0"
        ))

        # Show the specific equation with values
        # We reconstruct the equation string based on the loads
        equation_parts = []
        
        # Reaction B
        if other_support:
            dist = other_support.position - ref_support.position
            equation_parts.append(f"R_{{{other_support.support_id}}} \\cdot {dist:.2f}")
        
        # Loads
        rhs_value = 0.0
        
        for load in self.payload.point_loads:
            dist = load.position - ref_support.position
            if abs(dist) < 1e-6: continue
            angle_rad = math.radians(load.angle_deg)
            vertical = -load.magnitude * math.sin(angle_rad) # Upward component
            # Moment = r x F. 
            # Force is at 'dist'. Upward force creates CCW moment (+).
            # Downward force creates CW moment (-).
            
            sign = "+" if vertical > 0 else "-"
            equation_parts.append(f"{sign} {abs(vertical):.2f} \\cdot {dist:.2f}")
            rhs_value -= vertical * dist

        for udl in self.payload.udls:
            # Equivalent force
            span = udl.end - udl.start
            if udl.shape == "uniform":
                mag = udl.magnitude * span
                centroid = udl.start + span/2
            else:
                mag = udl.magnitude * span * 0.5 # Triangle
                centroid = udl.start + span * (2/3 if udl.shape == "triangular_increasing" else 1/3)
            
            dist = centroid - ref_support.position
            is_down = udl.direction == "down"
            # Down force -> CW moment (-)
            sign = "-" if is_down else "+"
            equation_parts.append(f"{sign} {mag:.2f} \\cdot {dist:.2f}")
            
            moment_contrib = -mag * dist if is_down else mag * dist
            rhs_value -= moment_contrib

        for moment in self.payload.moment_loads:
            is_ccw = moment.direction == "ccw"
            sign = "+" if is_ccw else "-"
            equation_parts.append(f"{sign} {moment.magnitude:.2f}")
            rhs_value -= (moment.magnitude if is_ccw else -moment.magnitude)

        full_eq = " ".join(equation_parts) + " = 0"
        
        steps.append(SolutionStep(
            step_number=4,
            title="Moment Denkleminin Çözümü",
            explanation="Oluşturulan moment denklemi çözülerek bilinmeyen mesnet tepkisi bulunur.",
            substituted_formula=full_eq,
            numerical_result=f"R_{{{other_support.support_id}}} = {other_support.vertical:.2f} \\text{{ kN}}" if other_support else "N/A"
        ))

        # Vertical Force Summation
        steps.append(SolutionStep(
            step_number=5,
            title="Düşey Kuvvet Dengesi",
            explanation="Toplam düşey kuvvetlerin sıfır olması şartından diğer mesnet tepkisi bulunur.",
            general_formula=r"\sum F_y = 0 \Rightarrow R_A + R_B - \sum P - \sum (q \cdot L) = 0"
        ))
        
        fy_parts = []
        fy_parts.append(f"R_{{{ref_support.support_id}}}")
        if other_support:
            fy_parts.append(f"+ {other_support.vertical:.2f}")
            
        # Add loads to string
        for load in self.payload.point_loads:
            angle_rad = math.radians(load.angle_deg)
            vertical = -load.magnitude * math.sin(angle_rad)
            sign = "+" if vertical > 0 else "-"
            fy_parts.append(f"{sign} {abs(vertical):.2f}")
            
        for udl in self.payload.udls:
            span = udl.end - udl.start
            mag = udl.magnitude * span * (0.5 if "triangular" in udl.shape else 1.0)
            is_down = udl.direction == "down"
            sign = "-" if is_down else "+"
            fy_parts.append(f"{sign} {mag:.2f}")
            
        fy_eq = " ".join(fy_parts) + " = 0"
        
        steps.append(SolutionStep(
            step_number=6,
            title="Düşey Denge Çözümü",
            explanation="Denklem çözülerek kalan mesnet tepkisi hesaplanır.",
            substituted_formula=fy_eq,
            numerical_result=f"R_{{{ref_support.support_id}}} = {ref_support.vertical:.2f} \\text{{ kN}}"
        ))

        return SolutionMethod(
            method_name="support_reactions",
            method_title="Mesnet Tepkileri",
            description="Statik denge denklemleri ile mesnet reaksiyonlarının hesaplanması.",
            steps=steps,
            recommended=True,
            recommendation_reason="Analize başlamadan önce dış mesnet tepkilerinin bulunması zorunludur."
        )

    def _solve_section_method(self) -> SolutionMethod:
        steps = []
        
        # Identify critical points (supports, loads)
        critical_points = sorted(list(set(
            [s.position for s in self.payload.supports] +
            [l.position for l in self.payload.point_loads] +
            [l.start for l in self.payload.udls] +
            [l.end for l in self.payload.udls] +
            [l.position for l in self.payload.moment_loads]
        )))
        
        # Add start and end if missing
        if 0 not in critical_points: critical_points.insert(0, 0)
        if self.payload.length not in critical_points: critical_points.append(self.payload.length)
        
        critical_points = sorted(list(set(critical_points)))

        steps.append(SolutionStep(
            step_number=1,
            title="Bölge Belirleme",
            explanation="Kiriş, yüklerin değiştiği noktalardan (tekil yükler, mesnetler, yayılı yük başlangıç/bitişleri) bölgelere ayrılır.",
            beam_section=BeamSectionHighlight(start=0, end=self.payload.length, label="Tüm Kiriş")
        ))

        for i in range(len(critical_points) - 1):
            start = critical_points[i]
            end = critical_points[i+1]
            mid = (start + end) / 2
            
            # Find shear and moment at mid point from diagram data
            # This is an approximation using the pre-calculated diagram
            # Ideally we would re-calculate symbolically
            
            # Find closest index in diagram x array
            x_vals = np.array([float(x) for x in self.diagram.x])
            idx = (np.abs(x_vals - mid)).argmin()
            
            shear_val = self.diagram.shear[idx]
            moment_val = self.diagram.moment[idx]

            steps.append(SolutionStep(
                step_number=len(steps) + 1,
                title=f"Bölge {i+1}: {start:.2f}m < x < {end:.2f}m",
                explanation=f"Bu aralıkta kiriş kesilerek iç kuvvetler yazılır. Soldan sağa x mesafesinde kesim yapılır.",
                general_formula=r"V(x) = \sum F_y, \quad M(x) = \sum M_{kesim}",
                substituted_formula=rf"V({mid:.2f}) \approx {shear_val} \\ M({mid:.2f}) \approx {moment_val}",
                beam_section=BeamSectionHighlight(start=start, end=end, label=f"Bölge {i+1}")
            ))

        return SolutionMethod(
            method_name="section_method",
            method_title="Kesme Yöntemi",
            description="Kirişin her yük değişiminde kesilerek iç kuvvet denklemlerinin çıkarılması.",
            steps=steps,
            recommended=False,
            recommendation_reason="Fonksiyonel denklemleri görmek için en iyi yöntemdir, ancak işlem yükü fazladır."
        )

    def _solve_area_method(self) -> SolutionMethod:
        steps = []
        
        steps.append(SolutionStep(
            step_number=1,
            title="Yöntem Prensibi",
            explanation="Kesme kuvveti diyagramının altındaki alan, moment değişimine eşittir. Moment diyagramının eğimi ise kesme kuvvetine eşittir.",
            general_formula=r"\Delta M = \int V(x) dx = \text{Kesme Diyagramı Alanı}"
        ))

        # Iterate through segments to calculate areas
        x_vals = [float(x) for x in self.diagram.x]
        shear_vals = [float(v) for v in self.diagram.shear]
        moment_vals = [float(v) for v in self.diagram.moment]
        
        # Identify zero crossings and critical points for area calculation
        # For simplicity, we use the diagram points directly but group them
        # A real implementation would integrate symbolically or numerically between critical points
        
        # We will use the same critical points as section method
        critical_points = sorted(list(set(
            [s.position for s in self.payload.supports] +
            [l.position for l in self.payload.point_loads] +
            [l.start for l in self.payload.udls] +
            [l.end for l in self.payload.udls] +
            [l.position for l in self.payload.moment_loads] + 
            [0, self.payload.length]
        )))
        critical_points = sorted(list(set(critical_points)))

        for i in range(len(critical_points) - 1):
            start = critical_points[i]
            end = critical_points[i+1]
            
            # Extract segment data
            indices = [j for j, x in enumerate(x_vals) if start <= x <= end]
            if not indices: continue
            
            segment_x = [x_vals[j] for j in indices]
            segment_shear = [shear_vals[j] for j in indices]
            segment_moment = [moment_vals[j] for j in indices]
            
            # Calculate area using trapezoidal rule
            area = np.trapz(segment_shear, segment_x)
            
            # Determine shape description
            # Check if shear is constant (rectangle) or linear (triangle/trapezoid)
            is_shear_constant = np.allclose(segment_shear, segment_shear[0], atol=1e-2)
            
            shape_desc = "Dikdörtgen Alan" if is_shear_constant else "Yamuk/Üçgen Alan"
            
            trend = "constant"
            if area > 0.1: trend = "increase"
            elif area < -0.1: trend = "decrease"

            steps.append(SolutionStep(
                step_number=len(steps) + 1,
                title=f"Aralık: {start:.2f}m - {end:.2f}m",
                explanation=f"Bu aralıktaki kesme kuvveti diyagramı alanı hesaplanır. {shape_desc} moment değişimini verir.",
                substituted_formula=rf"\Delta M = {area:.2f} \text{{ kN.m}}",
                numerical_result=f"M_{{son}} = M_{{baş}} + \Delta M = {moment_vals[0]:.2f} + {area:.2f} = {moment_vals[-1]:.2f}",
                area_visualization=AreaMethodVisualization(
                    shape=shape_desc,
                    area_value=area,
                    trend=trend,
                    region=ShearRegionSamples(x=segment_x, shear=segment_shear),
                    moment_segment=MomentSegmentSamples(x=segment_x, moment=segment_moment)
                )
            ))

        return SolutionMethod(
            method_name="area_method",
            method_title="Alan Yöntemi",
            description="Grafiksel çizim için en hızlı ve pratik yöntem.",
            steps=steps,
            recommended=True,
            recommendation_reason="Hızlı diyagram çizimi ve görsel kontrol için en uygun yöntemdir."
        )

    def _solve_differential_method(self) -> SolutionMethod:
        steps = []
        
        # Step 1: General Differential Relations
        steps.append(SolutionStep(
            step_number=1,
            title="Diferansiyel Bağıntılar",
            explanation="Yük, kesme kuvveti ve moment arasındaki türev ilişkileri kullanılır. Tekillik fonksiyonları (Macaulay parantezleri) kullanılarak tüm kiriş tek bir denklemle ifade edilir.",
            general_formula=r"\frac{dV}{dx} = -w(x), \quad \frac{dM}{dx} = V(x)"
        ))

        # Construct Load Function w(x)
        # We will build the LaTeX string manually for better control over engineering notation <x-a>^n
        
        w_terms = []
        v_terms = []
        m_terms = []

        # 1. Supports (Reactions)
        # Reactions are point loads. Upward reaction is positive load.
        # w(x) includes R * <x-a>^-1
        for reaction in self.reactions:
            pos = reaction.position
            mag = reaction.vertical
            if abs(mag) < 1e-6: continue
            
            # Load term: R * <x-a>^-1
            sign = "+" if mag >= 0 else "-"
            w_terms.append(f"{sign} {abs(mag):.2f} \\langle x - {pos:.2f} \\rangle^{{-1}}")
            
            # Shear term: Integral of -w. 
            # V_jump = +R. So V term is +R * <x-a>^0
            # Wait, dV/dx = -w. So V = - int w.
            # If w has +R (upward force), then -int w has -R.
            # BUT, standard convention: Upward force causes upward jump in Shear.
            # So V term should be +R * <x-a>^0.
            # This means our differential relation dV/dx = -w assumes w is downward positive?
            # Let's stick to: V(x) = Sum of forces to the left.
            # Upward force R at a: +R * <x-a>^0
            v_terms.append(f"{sign} {abs(mag):.2f} \\langle x - {pos:.2f} \\rangle^{{0}}")
            
            # Moment term: Integral of V.
            # M term: +R * <x-a>^1
            m_terms.append(f"{sign} {abs(mag):.2f} \\langle x - {pos:.2f} \\rangle^{{1}}")

            # Moment reaction (for fixed supports)
            if abs(reaction.moment) > 1e-6:
                # Reaction moment M_R. CCW is positive.
                # A concentrated moment M0 at a.
                # In V diagram: No effect (except via reactions).
                # In M diagram: Jump of -M0 (if CCW) or +M0?
                # Standard: CCW moment causes downward jump in Moment diagram?
                # Let's check: M(x) = Sum of moments to the left.
                # M0 (CCW) at a. For x > a, M0 is included.
                # Sum M = M0 + ... 
                # So term is +M0 * <x-a>^0.
                m_sign = "+" if reaction.moment >= 0 else "-"
                m_terms.append(f"{m_sign} {abs(reaction.moment):.2f} \\langle x - {pos:.2f} \\rangle^{{0}}")
                
                # In V: Moment load corresponds to doublet in w?
                # w term: M * <x-a>^-2
                # V term: M * <x-a>^-1 (Dirac in Shear?) - usually ignored in V diagram drawing but mathematically there.
                pass

        # 2. Point Loads
        for load in self.payload.point_loads:
            pos = load.position
            # Resolve vertical component
            angle_rad = math.radians(load.angle_deg)
            vertical = load.magnitude * math.sin(angle_rad) # Upward is positive y
            # But load magnitude is usually just force.
            # If angle is -90 (down), sin is -1. Vertical is -Mag.
            
            if abs(vertical) < 1e-6: continue
            
            # Upward force P: +P * <x-a>^0 in V.
            sign = "+" if vertical >= 0 else "-"
            v_terms.append(f"{sign} {abs(vertical):.2f} \\langle x - {pos:.2f} \\rangle^{{0}}")
            m_terms.append(f"{sign} {abs(vertical):.2f} \\langle x - {pos:.2f} \\rangle^{{1}}")
            
            # w term: P * <x-a>^-1
            w_terms.append(f"{sign} {abs(vertical):.2f} \\langle x - {pos:.2f} \\rangle^{{-1}}")

        # 3. UDLs
        for udl in self.payload.udls:
            start = udl.start
            end = udl.end
            mag = udl.magnitude
            is_down = udl.direction == "down"
            # Downward UDL: w = q.
            # Upward force is positive for V.
            # Downward load q: Force is -q per length.
            # V term: -q * <x-start>^1 + q * <x-end>^1
            
            eff_mag = -mag if is_down else mag
            sign = "+" if eff_mag >= 0 else "-"
            
            if udl.shape == "uniform":
                # w(x) = q <x-a>^0 - q <x-b>^0
                w_terms.append(f"{sign} {mag:.2f} \\langle x - {start:.2f} \\rangle^{{0}}")
                w_terms.append(f"{'-' if sign == '+' else '+'} {mag:.2f} \\langle x - {end:.2f} \\rangle^{{0}}")
                
                # V(x) = q <x-a>^1 - q <x-b>^1
                v_terms.append(f"{sign} {mag:.2f} \\langle x - {start:.2f} \\rangle^{{1}}")
                v_terms.append(f"{'-' if sign == '+' else '+'} {mag:.2f} \\langle x - {end:.2f} \\rangle^{{1}}")
                
                # M(x) = q/2 <x-a>^2 - q/2 <x-b>^2
                m_terms.append(f"{sign} {mag/2:.2f} \\langle x - {start:.2f} \\rangle^{{2}}")
                m_terms.append(f"{'-' if sign == '+' else '+'} {mag/2:.2f} \\langle x - {end:.2f} \\rangle^{{2}}")

        # 4. Moment Loads
        for moment in self.payload.moment_loads:
            pos = moment.position
            mag = moment.magnitude
            is_ccw = moment.direction == "ccw"
            # CCW Moment M0:
            # M term: +M0 * <x-a>^0
            sign = "+" if is_ccw else "-"
            m_terms.append(f"{sign} {mag:.2f} \\langle x - {pos:.2f} \\rangle^{{0}}")


        # Format equations
        def format_eq(terms):
            if not terms: return "0"
            s = " ".join(terms)
            # Clean up: "+ -" -> "- ", first "+" remove
            if s.startswith("+ "): s = s[2:]
            return s

        w_eq = format_eq(w_terms)
        v_eq = format_eq(v_terms)
        m_eq = format_eq(m_terms)

        steps.append(SolutionStep(
            step_number=2,
            title="Yük Fonksiyonu w(x)",
            explanation="Tüm dış yükler ve mesnet tepkileri tekillik fonksiyonları ile ifade edilir.",
            substituted_formula=f"w(x) = {w_eq}"
        ))
        
        steps.append(SolutionStep(
            step_number=3,
            title="Kesme Kuvveti Fonksiyonu V(x)",
            explanation="Yük fonksiyonunun entegrali alınarak kesme kuvveti denklemi elde edilir.",
            general_formula=r"V(x) = \int w(x) dx",
            substituted_formula=f"V(x) = {v_eq}"
        ))
        
        steps.append(SolutionStep(
            step_number=4,
            title="Moment Fonksiyonu M(x)",
            explanation="Kesme kuvveti fonksiyonunun entegrali alınarak moment denklemi elde edilir.",
            general_formula=r"M(x) = \int V(x) dx",
            substituted_formula=f"M(x) = {m_eq}"
        ))

        return SolutionMethod(
            method_name="differential_method",
            method_title="Diferansiyel Yöntem",
            description="Tekillik fonksiyonları (Macaulay Yöntemi) ile analitik çözüm.",
            steps=steps,
            recommended=False,
            recommendation_reason="Karmaşık yayılı yüklerde matematiksel kesinlik sağlar."
        )
