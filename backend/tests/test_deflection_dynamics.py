import math

import numpy as np
import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient

from beam_solver_backend.dynamics import simulate_beam
from beam_solver_backend.main import app
from beam_solver_backend.schemas import SimulationRequest, SolveRequest
from beam_solver_backend.solvers import solve_beam_unified


def request(**kwargs):
    base = dict(length=6, supports=[dict(id="A", type="pin", position=0), dict(id="B", type="roller", position=6)])
    return SimulationRequest(**(base | kwargs))


@pytest.mark.parametrize("angle, sign", [(-90, 1), (90, -1)])
def test_midspan_deflection_formula_and_support_conditions(angle, sign):
    p = request(point_loads=[dict(id="P", magnitude=10, position=3, angle_deg=angle)])
    result = solve_beam_unified(p)
    middle = result.diagram.x.index(3)
    expected_mm = sign*10*6**3/(48*200*1e6*1e-4)*1000
    assert result.diagram.deflection[middle] == pytest.approx(expected_mm, abs=1e-6)
    assert result.diagram.deflection[0] == pytest.approx(0, abs=1e-6)
    assert result.diagram.deflection[-1] == pytest.approx(0, abs=1e-6)


def test_udl_closed_form():
    p = request(udls=[dict(id="Q", magnitude=4, start=0, end=6)])
    r = solve_beam_unified(p)
    expected = 5*4*6**4/(384*200*1e6*1e-4)*1000
    assert max(r.diagram.deflection) == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize("fixed, tip", [(0, 6), (6, 0)])
def test_cantilever_tip_formula_and_rotation(fixed, tip):
    p = request(beam_type="cantilever", supports=[dict(id="A", type="fixed", position=fixed)],
                point_loads=[dict(id="P", magnitude=10, position=tip)])
    r = solve_beam_unified(p)
    i = r.diagram.x.index(fixed)
    j = r.diagram.x.index(tip)
    assert r.diagram.deflection[i] == pytest.approx(0, abs=1e-6)
    assert r.diagram.rotation[i] == pytest.approx(0, abs=1e-6)
    assert r.diagram.deflection[j] == pytest.approx(10*6**3/(3*200*1e6*1e-4)*1000, rel=1e-5)


@pytest.mark.parametrize("loads", [
    dict(point_loads=[dict(id="P", magnitude=10, position=3)]),
    dict(point_loads=[dict(id="P", magnitude=10, position=2.2, angle_deg=90)]),
    dict(moment_loads=[dict(id="M", magnitude=4, position=2, direction="ccw")]),
    dict(moment_loads=[dict(id="M", magnitude=4, position=0, direction="cw")]),
    *[dict(udls=[dict(id="Q", magnitude=4, start=1, end=5, shape=shape)]) for shape in
      ("uniform", "triangular_increasing", "triangular_decreasing")],
])
def test_independent_fem_static_limit_agrees_with_analytical_solver(loads):
    p = request(**loads)
    analytical = solve_beam_unified(p)
    simulation = simulate_beam(p)
    expected = np.interp(simulation.x, analytical.diagram.x, analytical.diagram.deflection)
    np.testing.assert_allclose(simulation.static_deflection_mm, expected, atol=4e-5, rtol=2e-4)
    # Every modal shape obeys the support boundary conditions for ALL time.
    assert np.max(np.abs(np.array(simulation.modal_static_shapes_mm)[:, [0, -1]])) < 1e-10


@pytest.mark.parametrize("beam_type, supports, beta", [
    ("simply_supported", [dict(id="A", type="pin", position=0), dict(id="B", type="roller", position=6)], math.pi),
    ("cantilever", [dict(id="A", type="fixed", position=0)], 1.87510407),
    ("cantilever", [dict(id="A", type="fixed", position=6)], 1.87510407),
])
def test_natural_frequency_against_continuous_beam_formula(beam_type, supports, beta):
    p = request(beam_type=beam_type, supports=supports)
    r = simulate_beam(p)
    expected = beta**2/(2*math.pi*6**2)*math.sqrt(200*1e9*1e-4/100)
    assert r.fundamental_frequency_hz == pytest.approx(expected, rel=1e-5)
    doubled_mass = simulate_beam(p.model_copy(update={"mass_per_length_kgm": 200}))
    assert doubled_mass.fundamental_frequency_hz == pytest.approx(r.fundamental_frequency_hz/math.sqrt(2), rel=1e-6)


def test_simulation_api_and_invalid_rigidity():
    with TestClient(app) as client:
        response = client.post("/api/simulate", json=request().model_dump())
        assert response.status_code == 200
        assert response.json()["element_count"] >= 40
        assert client.post("/api/simulate", json=request().model_dump() | {"mass_per_length_kgm": 0}).status_code == 422
    for field in ("elastic_modulus_gpa", "moment_inertia_m4"):
        with pytest.raises(ValidationError):
            SolveRequest(**(request().model_dump() | {field: 0}))


@pytest.mark.parametrize("shape", ["triangular_increasing", "triangular_decreasing"])
def test_partial_triangular_load_end_equilibrium(shape):
    p = request(udls=[dict(id="Q", magnitude=4, start=1, end=4, shape=shape)])
    r = solve_beam_unified(p)
    assert r.diagram.shear[-1] == pytest.approx(0, abs=1e-6)
    assert r.diagram.moment[-1] == pytest.approx(0, abs=1e-6)
    assert r.diagram.deflection[-1] == pytest.approx(0, abs=1e-6)


def test_vercel_entrypoint_reuses_same_application():
    import runpy
    from pathlib import Path
    entrypoint = runpy.run_path(str(Path(__file__).parents[1] / "main.py"))
    assert entrypoint["app"] is app
    with TestClient(entrypoint["app"]) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.post("/api/solve", json=request().model_dump()).status_code == 200
        assert client.post("/api/chimney/period", json=dict(height_m=60, elastic_modulus_gpa=30,
            moment_inertia_m4=0.6, mass_per_length_kgm=1800)).status_code == 200
