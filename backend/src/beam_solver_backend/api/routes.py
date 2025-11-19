from __future__ import annotations

from fastapi import APIRouter, HTTPException

from beam_solver_backend.schemas.beam import SolveRequest, SolveResponse
from beam_solver_backend.solver.static_solver import solve_beam
from beam_solver_backend.solver.cantilever_solver import solve_cantilever_beam

router = APIRouter()


@router.post("/solve", response_model=SolveResponse)
async def solve(payload: SolveRequest) -> SolveResponse:
    try:
        if payload.beam_type == "cantilever":
            return solve_cantilever_beam(payload)
        return solve_beam(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
