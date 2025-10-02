from __future__ import annotations

from pydantic import ValidationError
from pytest import approx, raises

from beam_solver_backend.schemas.beam import (
    MomentLoad,
    PointLoad,
    Sampling,
    SolveRequest,
    Support,
    UniformDistributedLoad,
)
from beam_solver_backend.solver.static_solver import solve_beam


def _supports(length: float):
    return [
        Support(id="A", type="pin", position=0.0),
        Support(id="B", type="roller", position=length),
    ]


def _reaction_map(result):
    return {reaction.support_id: reaction for reaction in result.reactions}


def test_uniform_udl_reactions_and_moment():
    request = SolveRequest(
        length=6.0,
        supports=_supports(6.0),
        udls=[UniformDistributedLoad(id="Q1", magnitude=5.0, start=0.0, end=6.0)],
        sampling=Sampling(points=301),
    )

    result = solve_beam(request)
    reactions = _reaction_map(result)

    assert reactions["A"].vertical == approx(15.0, rel=1e-3)
    assert reactions["B"].vertical == approx(15.0, rel=1e-3)
    assert reactions["A"].axial == approx(0.0, abs=1e-6)
    assert reactions["B"].axial == approx(0.0, abs=1e-6)
    assert max(result.diagram.moment) == approx(22.5, rel=5e-2)
    assert result.diagram.normal[-1] == approx(0.0, abs=1e-6)


def test_point_load_with_angle():
    request = SolveRequest(
        length=4.0,
        supports=_supports(4.0),
        point_loads=[PointLoad(id="P1", magnitude=12.0, position=1.5, angle_deg=-90.0)],
        sampling=Sampling(points=201),
    )

    result = solve_beam(request)
    reactions = _reaction_map(result)

    assert reactions["A"].vertical == approx(7.5, rel=1e-3)
    assert reactions["B"].vertical == approx(4.5, rel=1e-3)
    assert reactions["A"].axial == approx(0.0, abs=1e-6)
    assert reactions["B"].axial == approx(0.0, abs=1e-6)
    assert min(result.diagram.shear) == approx(-4.5, rel=1e-2)


def test_horizontal_load_generates_axial_reaction():
    request = SolveRequest(
        length=5.0,
        supports=_supports(5.0),
        point_loads=[PointLoad(id="H1", magnitude=8.0, position=2.0, angle_deg=0.0)],
        sampling=Sampling(points=151),
    )

    result = solve_beam(request)
    reactions = _reaction_map(result)

    assert reactions["A"].axial == approx(-8.0, rel=1e-3)
    assert reactions["B"].axial == approx(0.0, abs=1e-6)
    assert min(result.diagram.normal) == approx(-8.0, rel=1e-3)
    assert all(abs(value) < 1e-6 for value in result.diagram.shear)


def test_concentrated_moment_effect():
    request = SolveRequest(
        length=10.0,
        supports=_supports(10.0),
        moment_loads=[MomentLoad(id="M1", magnitude=80.0, position=0.0, direction="ccw")],
        sampling=Sampling(points=201),
    )

    result = solve_beam(request)
    reactions = _reaction_map(result)

    assert reactions["A"].vertical == approx(-8.0, rel=1e-3)
    assert reactions["B"].vertical == approx(8.0, rel=1e-3)
    assert max(result.diagram.moment) == approx(80.0, rel=1e-2)


def test_invalid_point_load_position():
    with raises(ValidationError):
        SolveRequest(
            length=3.0,
            supports=_supports(3.0),
            point_loads=[PointLoad(id="P", magnitude=5.0, position=3.5)],
        )


def test_invalid_support_count():
    with raises(ValidationError):
        SolveRequest(
            length=5.0,
            supports=[
                Support(id="A", type="pin", position=0.0),
                Support(id="B", type="roller", position=2.5),
                Support(id="C", type="roller", position=5.0),
            ],
        )

