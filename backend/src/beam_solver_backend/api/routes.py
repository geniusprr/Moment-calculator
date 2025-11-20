from __future__ import annotations

from fastapi import APIRouter, HTTPException

from beam_solver_backend.schemas.beam import SolveRequest, SolveResponse
from beam_solver_backend.schemas.chimney import ChimneyPeriodRequest, ChimneyPeriodResponse
from beam_solver_backend.solver.static_solver import solve_beam
from beam_solver_backend.solver.cantilever_solver import solve_cantilever_beam
from beam_solver_backend.solver.chimney import calculate_fundamental_period

router = APIRouter()


@router.post("/solve", response_model=SolveResponse)
async def solve(payload: SolveRequest) -> SolveResponse:
    try:
        if payload.beam_type == "cantilever":
            return solve_cantilever_beam(payload)
        return solve_beam(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/chimney/period", response_model=ChimneyPeriodResponse)
async def chimney_period(payload: ChimneyPeriodRequest) -> ChimneyPeriodResponse:
    return calculate_fundamental_period(payload)
