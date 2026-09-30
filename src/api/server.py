"""
FastAPI application: REST/SSE API + static web UI.

Run:
    uvicorn src.api.server:app --reload          (dev)
    uvicorn src.api.server:app --host 0.0.0.0    (self-hosted)
"""

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from config import BASE_DIR
from src.api.routes import chat, papers, research, system, topics
from src.llm.base import LLMError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("physicsrag")


def create_app() -> FastAPI:
    app = FastAPI(
        title="PhysicsRAG",
        version="1.0.0",
        description=(
            "Self-hosted RAG over an arXiv paper library, with an "
            "autonomous research loop and any LLM provider."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    for router in (
        system.router,
        papers.router,
        topics.router,
        chat.router,
        research.router,
    ):
        app.include_router(router)

    @app.exception_handler(LLMError)
    async def llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    web_dir = BASE_DIR / "web"
    if web_dir.exists():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
    else:
        logger.warning("web/ directory missing — API only, no UI served")

    return app


app = create_app()