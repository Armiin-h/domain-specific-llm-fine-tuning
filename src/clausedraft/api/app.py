"""HTTP API for one clause continuation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from clausedraft.api.examples import EXAMPLES
from clausedraft.config import REPO_ROOT
from clausedraft.eval.parse import parse_gallery
from clausedraft.infer.types import CLAUSE_TYPES

WEB_DIR = REPO_ROOT / "web"
GALLERY_PATH = REPO_ROOT / "docs" / "gallery.md"


class CompleteBody(BaseModel):
    clause_type: str
    prefix: str
    preset: str = "greedy"


def create_app(drafter, gallery_path: Path | None = None) -> FastAPI:
    app = FastAPI(title="ClauseDraft")
    saved_gallery = GALLERY_PATH if gallery_path is None else gallery_path

    @app.get("/api/health")
    def health() -> dict:
        return asdict(drafter.health())

    @app.get("/api/types")
    def types() -> dict:
        return {"types": list(CLAUSE_TYPES)}

    @app.get("/api/examples")
    def examples() -> dict:
        return {"examples": [asdict(item) for item in EXAMPLES]}

    @app.get("/api/gallery")
    def gallery() -> dict:
        if not saved_gallery.is_file():
            raise HTTPException(status_code=404, detail="gallery is not in the tree")
        try:
            report = parse_gallery(saved_gallery.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return asdict(report)

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
