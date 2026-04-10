from __future__ import annotations

import logging

from fastapi import FastAPI
from app.calendar_sync import router as calendar_sync_router

from app.auth_router import router as auth_router
from app.chat_router import router as chat_router
from app.ingest_router import router as ingest_router
from app.meditation_router import router as meditation_router
from app.mindfulness_router import router as mindfulness_router
from app.practice_router import router as practice_router

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

app = FastAPI(title="NeuroHeart Chat API", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"ok": True}


app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(ingest_router)
app.include_router(practice_router)
app.include_router(meditation_router)
app.include_router(mindfulness_router)
app.include_router(calendar_sync_router)
