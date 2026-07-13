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
from api.schemas import ClaimResult, EvidenceResult, SpanResult, VerifyRequest, VerifyResponse
from engine.calibration import load_artifact, validate_bindings
from engine.config import ConfigError, RerankerConfig, SignalModelConfig, load_config
from engine.decomposition import decompose, segment
from engine.gate import gate_decision, judge_claim
from engine.retrieval import EvidenceStore, Reranker, StoreError
from engine.retrieval.pipeline import retrieve_pool
from engine.retrieval.pool import EvidenceRef
from engine.retrieval.postgres import PostgresStore
from engine.retrieval.rerank import CrossEncoderReranker
from engine.signals import Scorer
from engine.signals.groundedness import LettuceDetectScorer

logger = logging.getLogger("plumb.api")

DEFAULT_CONFIG_PATH = "config/verifier.yaml"

ScorerFactory = Callable[[SignalModelConfig], Scorer]
RerankerFactory = Callable[[RerankerConfig], Reranker]


def _store_from_env() -> PostgresStore | None:
    """The tenant store connection is deployment config (ADR-0010): env-injected
    by the chart, read-only, and not part of the versioned engine behavior."""
    dsn = os.environ.get("PLUMB_STORE_DSN")
    if not dsn:
        return None
    table = os.environ.get("PLUMB_STORE_TABLE")
    id_column = os.environ.get("PLUMB_STORE_ID_COLUMN")
    text_column = os.environ.get("PLUMB_STORE_TEXT_COLUMN")
    if not (table and id_column and text_column):
        raise ConfigError(
            "PLUMB_STORE_DSN is set but the store schema is incomplete — "
            "PLUMB_STORE_TABLE, PLUMB_STORE_ID_COLUMN, and PLUMB_STORE_TEXT_COLUMN "
            "are all required"
        )
    return PostgresStore(
        dsn=dsn,
        table=table,
        id_column=id_column,
        text_column=text_column,
        source_column=os.environ.get("PLUMB_STORE_SOURCE_COLUMN"),
        snapshot_column=os.environ.get("PLUMB_STORE_SNAPSHOT_COLUMN"),
        regconfig=os.environ.get("PLUMB_STORE_REGCONFIG", "simple"),
    )


def create_app(
    config_path: str | Path | None = None,
    scorer_factory: ScorerFactory | None = None,
    store: EvidenceStore | None = None,
    reranker_factory: RerankerFactory | None = None,
) -> FastAPI:
    setup_logging()
    resolved_config = Path(config_path or os.environ.get("PLUMB_CONFIG", DEFAULT_CONFIG_PATH))
    cfg = load_config(resolved_config)
    # The artifact travels with the config; its path resolves against the config
    # file's directory. A missing or mismatched calibrator is a startup failure —
    # the engine never serves raw scores (ADR-0008).
    artifact = load_artifact(resolved_config.parent / cfg.groundedness.calibration)
    validate_bindings(artifact, cfg.groundedness)
    engine_version = package_version("plumb")
    factory: ScorerFactory = scorer_factory or LettuceDetectScorer.load
    resolved_store = store if store is not None else _store_from_env()
    # Thorough mode needs both halves: the versioned retrieval knobs and a
    # deployed store connection. Anything less is a fast-only deployment.
    thorough_enabled = cfg.retrieval is not None and resolved_store is not None
    load_reranker: RerankerFactory = reranker_factory or CrossEncoderReranker.load

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "loading scoring model",
            extra={"model": cfg.groundedness.model, "revision": cfg.groundedness.revision},
        )
        app.state.scorer = factory(cfg.groundedness)
        logger.info("scoring model ready")
        if thorough_enabled:
            assert cfg.retrieval is not None  # thorough_enabled implies it
            probe = getattr(resolved_store, "probe", None)
            if probe is not None:
                probe()
                logger.info("tenant store probe succeeded")
            logger.info(
                "loading reranker",
                extra={
                    "model": cfg.retrieval.reranker.model,
                    "revision": cfg.retrieval.reranker.revision,
                },
            )
            app.state.reranker = load_reranker(cfg.retrieval.reranker)
            logger.info("reranker ready")
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

    @app.post("/v1/verify", response_model_exclude_none=True)
    def verify(request: VerifyRequest) -> VerifyResponse:
        # Sync on purpose: inference blocks, so FastAPI must run this in its
        # threadpool — `async def` here would serialize the event loop.
        if request.mode not in ("fast", "thorough"):
            raise HTTPException(
                status_code=400,
                detail=f"mode {request.mode!r} is not supported — "
                "supported modes are 'fast' and 'thorough'",
            )
        scorer: Scorer = app.state.scorer
        evidence: list[list[EvidenceRef]] | None = None
        if request.mode == "fast":
            if not request.context:
                raise HTTPException(
                    status_code=400,
                    detail="context is required and must be non-empty in fast mode",
                )
            passages = request.context
        else:
            if not thorough_enabled:
                raise HTTPException(
                    status_code=400,
                    detail="this deployment is fast-only: thorough mode needs a "
                    "tenant store connection and a retrieval config section",
                )
            assert cfg.retrieval is not None and resolved_store is not None
            claim_texts = [claim.text for claim in segment(request.text)]
            try:
                pool = retrieve_pool(
                    claim_texts,
                    request.context or [],
                    resolved_store,
                    app.state.reranker,
                    cfg.retrieval,
                    scorer.count_tokens,
                )
            except StoreError as exc:
                # Never a verdict on partial evidence: an erroring store fails
                # the verification, loudly.
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            if not pool.passages:
                raise HTTPException(
                    status_code=422,
                    detail="no evidence to verify against: retrieval returned "
                    "nothing for any claim's query and the request carried no context",
                )
            passages = pool.passages
            evidence = pool.evidence
        scores = scorer.score(request.text, passages)
        if evidence is not None and scores.truncated:
            logger.error(
                "scorer truncated the context in thorough mode — the pool budget "
                "should have prevented this; check pool_budget_tokens against the "
                "model window",
                extra={"passage_count": len(passages)},
            )
        assessed = decompose(request.text, scores, cfg.groundedness.span_threshold)
        confidences = [artifact.confidence(claim.support) for claim in assessed]
        span_confidences = [
            [artifact.span_confidence(span.raw_risk) for span in claim.spans] for claim in assessed
        ]
        verdicts = [
            judge_claim(claim.text, confidence, cfg.groundedness.threshold)
            for claim, confidence in zip(assessed, confidences, strict=True)
        ]
        # Raw numbers are log-only detail: the response carries the calibrated
        # claim and span confidences, and anyone thresholding must threshold those.
        logger.info(
            "claims calibrated",
            extra={
                "claims": [
                    {
                        "start": claim.start,
                        "end": claim.end,
                        "raw_support": claim.support,
                        "confidence": confidence,
                        "spans": [
                            {
                                "start": span.start,
                                "end": span.end,
                                "raw_risk": span.raw_risk,
                                "confidence": span_confidence,
                            }
                            for span, span_confidence in zip(
                                claim.spans, claim_span_confidences, strict=True
                            )
                        ],
                    }
                    for claim, confidence, claim_span_confidences in zip(
                        assessed, confidences, span_confidences, strict=True
                    )
                ]
            },
        )
        if evidence is not None and len(evidence) != len(assessed):
            raise RuntimeError(
                f"retrieval saw {len(evidence)} claims but decomposition produced "
                f"{len(assessed)} — the segmenter must be deterministic"
            )
        return VerifyResponse(
            claims=[
                ClaimResult(
                    text=claim.text,
                    start=claim.start,
                    end=claim.end,
                    verdict=verdict.verdict,
                    confidence=verdict.confidence,
                    spans=[
                        SpanResult(
                            start=span.start,
                            end=span.end,
                            text=span.text,
                            confidence=span_confidence,
                        )
                        for span, span_confidence in zip(
                            claim.spans, claim_span_confidences, strict=True
                        )
                    ],
                    evidence=None
                    if evidence is None
                    else [
                        EvidenceResult(
                            source_id=ref.source_id,
                            chunk_id=ref.chunk_id,
                            rank=ref.rank,
                            snapshot_id=ref.snapshot_id,
                        )
                        for ref in evidence[i]
                    ],
                )
                for i, (claim, verdict, claim_span_confidences) in enumerate(
                    zip(assessed, verdicts, span_confidences, strict=True)
                )
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
        if thorough_enabled and getattr(app.state, "reranker", None) is None:
            return JSONResponse(status_code=503, content={"status": "loading reranker"})
        return JSONResponse(status_code=200, content={"status": "ready"})

    return app
