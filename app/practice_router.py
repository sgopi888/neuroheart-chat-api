from __future__ import annotations

import datetime
import logging
from typing import Optional

from fastapi import APIRouter, Header, HTTPException

from app.chat_service import generate_practice_script
from app.config import settings
from app.schemas import PracticeRequest, PracticeResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/practice", tags=["practice"])


def _require_app_token(x_app_token: Optional[str]) -> None:
    if settings.app_token and x_app_token != settings.app_token:
        raise HTTPException(status_code=403, detail="forbidden")


@router.post("/generate", response_model=PracticeResponse)
async def generate_practice(
    req: PracticeRequest,
    x_app_token: Optional[str] = Header(default=None),
) -> dict:
    _require_app_token(x_app_token)

    try:
        script = await generate_practice_script(
            user_uid=req.user_uid,
            conversation_id=req.conversation_id,
            mood=req.mood,
            depth=req.depth,
            duration=req.duration,
            session_type=req.session_type,
        )

        today = datetime.date.today().isoformat()
        title = f"Practice: {req.mood} {today}"

        return {
            "conversation_id": req.conversation_id,
            "script": script,
            "title": title,
        }
    except LookupError:
        raise HTTPException(status_code=404, detail="conversation_not_found")
    except Exception as exc:
        logger.exception("generate_practice failed: %s", exc)
        raise HTTPException(status_code=500, detail="practice_generation_failed")
