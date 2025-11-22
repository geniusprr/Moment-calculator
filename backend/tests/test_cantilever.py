from __future__ import annotations

from pytest import approx

from beam_solver_backend.schemas.beam import PointLoad, SolveRequest, Support
from beam_solver_backend.solver.cantilever_solver import solve_cantilever_beam


def test_cantilever_tip_load_reactions_and_moment():
    request = SolveRequest(
        length=5.0,
        beam_type="cantilever",
        supports=[Support(id="A", type="fixed", position=0.0)],
        point_loads=[PointLoad(id="P1", magnitude=10.0, position=5.0, angle_deg=-90.0)],
    )

    result = solve_cantilever_beam(request)
    reaction = result.reactions[0]

    assert reaction.vertical == approx(10.0, rel=1e-3)
    assert reaction.axial == approx(0.0, abs=1e-6)
    assert reaction.moment == approx(-50.0, rel=1e-3)
    assert min(result.diagram.moment) == approx(-50.0, rel=1e-3)
    assert result.diagram.moment[-1] == approx(0.0, abs=1e-6)

