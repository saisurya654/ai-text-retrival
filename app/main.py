from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router
from app.utils.logging import configure_logging

configure_logging()

app = FastAPI(
    title="LegalMind AI",
    description="Grounded legal document understanding and drafting platform.",
    version="1.0.0",
)
app.include_router(router)


@app.get("/health")
def healthcheck():
    return {"status": "ok"}
