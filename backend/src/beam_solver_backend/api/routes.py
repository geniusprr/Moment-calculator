from __future__ import annotations

from fastapi import APIRouter, HTTPException

from beam_solver_backend.schemas.beam import SolveRequest, SolveResponse
from beam_solver_backend.solver.static_solver import solve_beam

router = APIRouter()


@router.post("/solve", response_model=SolveResponse)
async def solve(payload: SolveRequest) -> SolveResponse:
    try:
        return solve_beam(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
