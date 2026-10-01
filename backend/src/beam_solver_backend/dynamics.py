"""Linear Euler–Bernoulli FEM with consistent mass and analytical modal response.

SI units internally. Positive transverse displacement is downward; nodal
rotation is dw/dx. The client evaluates the closed-form damped step response
of EVERY unconstrained mode, starting from zero displacement and velocity.
"""
from __future__ import annotations

import math
import numpy as np

from beam_solver_backend.schemas import SimulationRequest, SimulationResponse


def hermite(r: float, h: float) -> np.ndarray:
    return np.array([1 - 3*r*r + 2*r**3, h*(r - 2*r*r + r**3),
                     3*r*r - 2*r**3, h*(-r*r + r**3)])


def simulate_beam(payload: SimulationRequest) -> SimulationResponse:
    # Make every load discontinuity and support a node. Merge near-equal
    # coordinates to avoid tiny elements and ill-conditioned eigensystems.
    critical = [0.0, payload.length]
    critical += [s.position for s in payload.supports]
    critical += [p.position for p in payload.point_loads]
    critical += [p.position for p in payload.moment_loads]
    critical += [v for q in payload.udls for v in (q.start, q.end)]
    critical = sorted(set(critical))
    if any(b - a < payload.length * 1e-7 for a, b in zip(critical, critical[1:])):
        raise ValueError("Simülasyon için farklı yük/mesnet konumlarını biraz ayırın; çok yakın noktalar sayısal kararlılığı bozuyor.")
    coordinates = [critical[0]]
    for a, b in zip(critical, critical[1:]):
        n = max(1, math.ceil((b-a) / (payload.length / 40)))
        coordinates.extend(np.linspace(a, b, n+1)[1:].tolist())
    nodes = np.array(coordinates)
    ndof = 2 * len(nodes)
    stiffness = np.zeros((ndof, ndof))
    mass = np.zeros((ndof, ndof))
    force = np.zeros(ndof)
    ei = payload.elastic_modulus_gpa * 1e9 * payload.moment_inertia_m4
    gauss_r, gauss_w = np.polynomial.legendre.leggauss(4)
    for i, (a, b) in enumerate(zip(nodes, nodes[1:])):
        h = b-a
        dofs = np.arange(2*i, 2*i+4)
        ke = ei/h**3 * np.array([
            [12, 6*h, -12, 6*h], [6*h, 4*h*h, -6*h, 2*h*h],
            [-12, -6*h, 12, -6*h], [6*h, 2*h*h, -6*h, 4*h*h]])
        me = payload.mass_per_length_kgm*h/420 * np.array([
            [156, 22*h, 54, -13*h], [22*h, 4*h*h, 13*h, -3*h*h],
            [54, 13*h, 156, -22*h], [-13*h, -3*h*h, -22*h, 4*h*h]])
        stiffness[np.ix_(dofs, dofs)] += ke
        mass[np.ix_(dofs, dofs)] += me
        for q in payload.udls:
            if a >= q.start-1e-10 and b <= q.end+1e-10:
                for gr, gw in zip(gauss_r, gauss_w):
                    r = (gr+1)/2
                    x = a+r*h
                    shape = 1.0
                    if q.shape == "triangular_increasing":
                        shape = (x-q.start)/(q.end-q.start)
                    elif q.shape == "triangular_decreasing":
                        shape = (q.end-x)/(q.end-q.start)
                    intensity = q.magnitude * 1000 * shape * (1 if q.direction == "down" else -1)
                    force[dofs] += hermite(r, h) * intensity * gw*h/2

    def node_at(position: float) -> int:
        return int(np.argmin(np.abs(nodes-position)))

    for p in payload.point_loads:
        force[2*node_at(p.position)] += -p.magnitude*1000*math.sin(math.radians(p.angle_deg))
    for p in payload.moment_loads:
        # Downward-positive slope is clockwise; CCW couples have negative work.
        force[2*node_at(p.position)+1] += p.magnitude*1000*(-1 if p.direction == "ccw" else 1)
    constrained = set()
    for support in payload.supports:
        j = 2*node_at(support.position)
        constrained.add(j)
        if support.type == "fixed":
            constrained.add(j+1)
    free = np.array([j for j in range(ndof) if j not in constrained])
    k = stiffness[np.ix_(free, free)]
    m = mass[np.ix_(free, free)]
    try:
        lower = np.linalg.cholesky(m)
        lower_inv = np.linalg.solve(lower, np.eye(len(free)))
        reduced = lower_inv @ k @ lower_inv.T
        eigenvalues, eigenvectors = np.linalg.eigh((reduced+reduced.T)/2)
        if np.any(eigenvalues <= 0):
            raise ValueError("Mesnet sistemi titreşim hesabı için kararlı değil.")
        modes = np.linalg.solve(lower.T, eigenvectors)
        static_amplitudes = (modes.T @ force[free]) / eigenvalues
    except np.linalg.LinAlgError as exc:
        raise ValueError("Kiriş modeli sayısal olarak çözülemedi. Mesnetleri ve rijitliği kontrol edin.") from exc
    full_modes = np.zeros((ndof, len(free)))
    full_modes[free] = modes * static_amplitudes
    # Hermite interpolation retains slopes, rather than a polygon at FE nodes.
    x_axis = np.unique(np.concatenate((np.linspace(0, payload.length, 241), nodes)))
    shapes = np.zeros((len(free), len(x_axis)))
    for j, x in enumerate(x_axis):
        i = min(int(np.searchsorted(nodes, x, side="right")-1), len(nodes)-2)
        h = nodes[i+1]-nodes[i]
        shapes[:, j] = hermite((x-nodes[i])/h, h) @ full_modes[2*i:2*i+4] * 1000
    return SimulationResponse(
        x=x_axis.tolist(), static_deflection_mm=shapes.sum(axis=0).tolist(),
        angular_frequencies_rad_s=np.sqrt(eigenvalues).tolist(),
        modal_static_shapes_mm=shapes.tolist(), damping_ratio=payload.damping_ratio,
        fundamental_frequency_hz=float(np.sqrt(eigenvalues[0])/(2*math.pi)),
        element_count=len(nodes)-1,
        notes=["Doğrusal elastik Euler–Bernoulli kiriş; küçük yer değiştirme ve sabit E, I varsayımı.",
               "Tutarlı kütle matrisi ve tüm serbest modlarla, ani uygulanan sabit yükün sönümlü tepkisi.",
               "Kütle kg/m cinsindedir; öz ağırlık yük olarak otomatik eklenmez.",
               "Kesme deformasyonu, çatlama, plastisite, burkulma ve mesnet ayrılması modellenmez."])
