"""Uvicorn entry point: `uvicorn api.main:app` (config path via PLUMB_CONFIG)."""

from api.app import create_app

app = create_app()
