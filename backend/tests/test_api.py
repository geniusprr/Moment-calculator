from __future__ import annotations

import httpx
import pytest

from beam_solver_backend.main import create_app


@pytest.mark.asyncio
async def test_solve_endpoint_success():
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    payload = {
        "length": 5.0,
        "supports": [
            {"id": "A", "type": "pin", "position": 0.0},
            {"id": "B", "type": "roller", "position": 5.0},
        ],
        "point_loads": [
            {"id": "P1", "magnitude": 10.0, "position": 2.5, "angle_deg": -90.0}
        ],
        "udls": [],
        "moment_loads": [],
        "sampling": {"points": 201},
    }

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/solve", json=payload)

    assert response.status_code == 200
    data = response.json()

    reactions = {item["support_id"]: item for item in data["reactions"]}
    assert reactions["A"]["vertical"] == pytest.approx(5.0, rel=1e-2)
    assert reactions["B"]["vertical"] == pytest.approx(5.0, rel=1e-2)
    assert reactions["A"]["axial"] == pytest.approx(0.0, abs=1e-6)
    assert "normal" in data["diagram"]
    assert data["diagram"]["moment"][0] == pytest.approx(0.0, abs=1e-6)
    assert data["meta"]["recommendation"]["method"] == "area"


@pytest.mark.asyncio
async def test_solve_endpoint_validation_error():
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    payload = {
        "length": 2.0,
        "supports": [
            {"id": "A", "type": "pin", "position": 0.0},
            {"id": "B", "type": "roller", "position": 2.0},
        ],
        "point_loads": [
            {"id": "P1", "magnitude": 10.0, "position": 3.0, "angle_deg": -90.0}
        ],
        "udls": [],
    }

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/solve", json=payload)

    assert response.status_code == 422
