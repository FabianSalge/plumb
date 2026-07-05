"""HTTP surface: /v1/verify, health probes, validation, version stamping."""

import logging
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from importlib.metadata import version as package_version
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.logging import RequestLoggingMiddleware, setup_logging
from api.schemas import ClaimResult, VerifyRequest, VerifyResponse
from engine.config import SignalModelConfig, load_config
from engine.scoring import HHEMScorer, Scorer
from engine.verdict import gate_decision, judge_claim

logger = logging.getLogger("plumb.api")

DEFAULT_CONFIG_PATH = "config/verifier.yaml"

ScorerFactory = Callable[[SignalModelConfig], Scorer]


def create_app(
    config_path: str | Path | None = None,
    scorer_factory: ScorerFactory | None = None,
) -> FastAPI:
    setup_logging()
    cfg = load_config(config_path or os.environ.get("PLUMB_CONFIG", DEFAULT_CONFIG_PATH))
    engine_version = package_version("plumb")
    factory: ScorerFactory = scorer_factory or HHEMScorer.load

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "loading scoring model",
            extra={"model": cfg.groundedness.model, "revision": cfg.groundedness.revision},
        )
        app.state.scorer = factory(cfg.groundedness)
        logger.info("scoring model ready")
        yield

    app = FastAPI(title="plumb", version=engine_version, lifespan=lifespan)
    app.add_middleware(RequestLoggingMiddleware)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        detail = [
            {
                "field": ".".join(str(part) for part in err["loc"] if part != "body"),
                "message": err["msg"],
            }
            for err in exc.errors()
        ]
        return JSONResponse(status_code=400, content={"error": "invalid request", "detail": detail})

    @app.post("/v1/verify")
    def verify(request: VerifyRequest) -> VerifyResponse:
        # Sync on purpose: inference blocks, so FastAPI must run this in its
        # threadpool — `async def` here would serialize the event loop.
        if request.mode != "fast":
            raise HTTPException(
                status_code=400,
                detail=f"mode {request.mode!r} is not supported — only 'fast' is available",
            )
        scorer: Scorer = app.state.scorer
        scores = scorer.score(request.text, request.context)
        claims = [judge_claim(request.text, scores, cfg.groundedness.threshold)]
        return VerifyResponse(
            claims=[
                ClaimResult(
                    text=claim.text,
                    verdict=claim.verdict,
                    score=claim.score,
                    evidence_index=claim.evidence_index,
                )
                for claim in claims
            ],
            gate=gate_decision(claims),
            engine_version=engine_version,
            config_version=cfg.version,
        )

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> JSONResponse:
        if getattr(app.state, "scorer", None) is None:
            return JSONResponse(status_code=503, content={"status": "loading model"})
        return JSONResponse(status_code=200, content={"status": "ready"})

    return app
