"""HTTP API for one clause continuation."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from clausedraft.api.examples import EXAMPLES
from clausedraft.config import REPO_ROOT
from clausedraft.infer.types import CLAUSE_TYPES

WEB_DIR = REPO_ROOT / "web"


class CompleteBody(BaseModel):
    clause_type: str
    prefix: str
    preset: str = "greedy"


def create_app(drafter) -> FastAPI:
    app = FastAPI(title="ClauseDraft")

    @app.get("/api/health")
    def health() -> dict:
        return asdict(drafter.health())

    @app.get("/api/types")
    def types() -> dict:
        return {"types": list(CLAUSE_TYPES)}

    @app.get("/api/examples")
    def examples() -> dict:
        return {"examples": [asdict(item) for item in EXAMPLES]}

    @app.post("/api/complete")
    def complete(body: CompleteBody) -> dict:
        try:
            result = drafter.complete(body.clause_type, body.prefix, body.preset)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return asdict(result)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    if (WEB_DIR / "app.js").is_file():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app
