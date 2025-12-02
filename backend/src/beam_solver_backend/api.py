from __future__ import annotations

from fastapi import APIRouter, HTTPException

from beam_solver_backend.schemas import (
    ChimneyPeriodRequest,
    ChimneyPeriodResponse,
    SolveRequest,
    SolveResponse,
)
from beam_solver_backend.solvers import (
    calculate_fundamental_period,
    solve_beam,
    solve_cantilever_beam,
)

router = APIRouter()


@router.post("/solve", response_model=SolveResponse)
async def solve(payload: SolveRequest) -> SolveResponse:
    """Solve either a simply supported or cantilever beam based on the request."""
    try:
        if payload.beam_type == "cantilever":
            return solve_cantilever_beam(payload)
        return solve_beam(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/chimney/period", response_model=ChimneyPeriodResponse)
async def chimney_period(payload: ChimneyPeriodRequest) -> ChimneyPeriodResponse:
    """Return the first-mode dynamic period for a slender chimney."""
    return calculate_fundamental_period(payload)
