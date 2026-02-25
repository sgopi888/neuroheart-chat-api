from __future__ import annotations

import logging

from fastapi import FastAPI

from app.chat_router import router as chat_router

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

app = FastAPI(title="NeuroHeart Chat API", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"ok": True}


app.include_router(chat_router)
