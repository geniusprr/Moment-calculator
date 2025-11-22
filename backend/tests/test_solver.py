from __future__ import annotations

from pydantic import ValidationError
from pytest import approx, raises

from beam_solver_backend.schemas.beam import (
    MomentLoad,
    PointLoad,
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
    )

    result = solve_beam(request)
    reactions = _reaction_map(result)

    assert reactions["A"].vertical == approx(15.0, rel=1e-3)
    assert reactions["B"].vertical == approx(15.0, rel=1e-3)
    assert reactions["A"].axial == approx(0.0, abs=1e-6)
    assert reactions["B"].axial == approx(0.0, abs=1e-6)
    assert max(result.diagram.moment) == approx(22.5, rel=5e-2)
    assert result.diagram.normal[-1] == approx(0.0, abs=1e-6)
    assert result.meta.recommendation.method == "area"


def test_point_load_with_angle():
    request = SolveRequest(
        length=4.0,
        supports=_supports(4.0),
        point_loads=[PointLoad(id="P1", magnitude=12.0, position=1.5, angle_deg=-90.0)],
    )

    result = solve_beam(request)
    reactions = _reaction_map(result)

    assert reactions["A"].vertical == approx(7.5, rel=1e-3)
    assert reactions["B"].vertical == approx(4.5, rel=1e-3)
    assert reactions["A"].axial == approx(0.0, abs=1e-6)
    assert reactions["B"].axial == approx(0.0, abs=1e-6)
    assert min(result.diagram.shear) == approx(-4.5, rel=1e-2)
    assert result.meta.recommendation.method == "area"


def test_horizontal_load_generates_axial_reaction():
    request = SolveRequest(
        length=5.0,
        supports=_supports(5.0),
        point_loads=[PointLoad(id="H1", magnitude=8.0, position=2.0, angle_deg=0.0)],
    )

    result = solve_beam(request)
    reactions = _reaction_map(result)

    assert reactions["A"].axial == approx(-8.0, rel=1e-3)
    assert reactions["B"].axial == approx(0.0, abs=1e-6)
    assert min(result.diagram.normal) == approx(-8.0, rel=1e-3)
    assert all(abs(value) < 1e-6 for value in result.diagram.shear)
    # Simplified solver always recommends area method
    assert result.meta.recommendation.method == "area"


def test_concentrated_moment_effect():
    request = SolveRequest(
        length=10.0,
        supports=_supports(10.0),
        moment_loads=[MomentLoad(id="M1", magnitude=80.0, position=0.0, direction="ccw")],
    )

    result = solve_beam(request)
    reactions = _reaction_map(result)

    assert reactions["A"].vertical == approx(-8.0, rel=1e-3)
    assert reactions["B"].vertical == approx(8.0, rel=1e-3)
    assert max(result.diagram.moment) == approx(80.0, rel=1e-2)
    assert result.meta.recommendation.method == "area"


def test_triangular_increasing_distributed_load():
    request = SolveRequest(
        length=6.0,
        supports=_supports(6.0),
        udls=[
            UniformDistributedLoad(
                id="Qtri",
                magnitude=6.0,
                start=0.0,
                end=6.0,
                direction="down",
                shape="triangular_increasing",
            )
        ],
    )

    result = solve_beam(request)
    reactions = _reaction_map(result)

    assert reactions["A"].vertical == approx(6.0, rel=1e-3)
    assert reactions["B"].vertical == approx(12.0, rel=1e-3)
    # Simplified solver always recommends area method
    assert result.meta.recommendation.method == "area"
    assert max(result.diagram.moment) == approx(13.86, rel=1e-1)


def test_area_method_recommended_for_vertical_point_loads():
    request = SolveRequest(
        length=8.0,
        supports=_supports(8.0),
        point_loads=[
            PointLoad(id="P1", magnitude=10.0, position=2.0, angle_deg=-90.0),
            PointLoad(id="P2", magnitude=5.0, position=6.0, angle_deg=-90.0),
        ],
    )

    result = solve_beam(request)

    assert result.meta.recommendation.method == "area"
    # Updated reason text
    assert "standart" in result.meta.recommendation.reason.lower()


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


def test_moment_extrema_metadata_with_partial_udl():
    request = SolveRequest(
        length=6.0,
        supports=_supports(6.0),
        udls=[UniformDistributedLoad(id="Q1", magnitude=16.0, start=0.0, end=4.0)],
        point_loads=[PointLoad(id="P1", magnitude=20.0, position=6.0, angle_deg=-90.0)],
    )

    result = solve_beam(request)

    assert result.meta.max_positive_moment == approx(512 / 9, rel=1e-4)
    assert result.meta.max_positive_position == approx(8 / 3, rel=1e-4)
    assert result.meta.max_absolute_moment == approx(512 / 9, rel=1e-4)
    assert result.meta.max_absolute_position == approx(8 / 3, rel=1e-4)
    assert result.meta.min_negative_moment is None


def test_reference_case_with_applied_moment_and_overhang():
    request = SolveRequest(
        length=6.0,
        supports=[
            Support(id="A", type="pin", position=0.0),
            Support(id="B", type="roller", position=4.0),
        ],
        udls=[UniformDistributedLoad(id="Q1", magnitude=16.0, start=0.0, end=4.0)],
        point_loads=[PointLoad(id="P1", magnitude=20.0, position=6.0, angle_deg=-90.0)],
        moment_loads=[MomentLoad(id="M1", magnitude=16.0, position=0.0, direction="cw")],
    )

    result = solve_beam(request)
    reactions = _reaction_map(result)

    assert reactions["A"].vertical == approx(26.0, rel=1e-6)
    assert reactions["B"].vertical == approx(58.0, rel=1e-6)
    assert result.meta.max_positive_position == approx(1.625, rel=1e-6)
    assert result.meta.max_positive_moment == approx(5.125, rel=1e-6)

