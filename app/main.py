from __future__ import annotations

from fastapi import FastAPI

from app.chat_router import router as chat_router

app = FastAPI(title="NeuroHeart Chat API", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"ok": True}


app.include_router(chat_router)
