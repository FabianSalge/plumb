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
from api.schemas import ClaimResult, SpanResult, VerifyRequest, VerifyResponse
from engine.config import SignalModelConfig, load_config
from engine.decomposition import decompose
from engine.scoring import LettuceDetectScorer, Scorer
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
    factory: ScorerFactory = scorer_factory or LettuceDetectScorer.load

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
        assessed = decompose(request.text, scores, cfg.groundedness.span_threshold)
        verdicts = [
            judge_claim(claim.text, claim.support, cfg.groundedness.threshold) for claim in assessed
        ]
        return VerifyResponse(
            claims=[
                ClaimResult(
                    text=claim.text,
                    start=claim.start,
                    end=claim.end,
                    verdict=verdict.verdict,
                    score=verdict.score,
                    spans=[
                        SpanResult(start=span.start, end=span.end, text=span.text)
                        for span in claim.spans
                    ],
                )
                for claim, verdict in zip(assessed, verdicts, strict=True)
            ],
            gate=gate_decision(verdicts),
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
