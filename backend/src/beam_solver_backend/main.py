from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from beam_solver_backend.api import router as beam_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI instance for the solver backend."""
    app = FastAPI(title="Beam Solver API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(beam_router, prefix="/api", tags=["solver"])

    @app.get("/health", tags=["health"])  # basit sağlık kontrolü
    async def health():
        """Return a minimal health payload for uptime checks."""
        return {"status": "ok"}

    return app


app = create_app()
