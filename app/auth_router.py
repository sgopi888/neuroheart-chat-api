"""
Auth endpoints: register, login, profile.

Uses Apple Sign In for authentication. User profile data (name, email,
age_range) is collected on first registration and stored in the users table.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from app.auth import verify_apple_token
from app.config import settings
from app.db import get_engine
from sqlalchemy import text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/auth", tags=["auth"])


# ---- Schemas ----

class RegisterRequest(BaseModel):
    apple_id_token: str
    name: Optional[str] = None
    email: Optional[str] = None
    age_range: Optional[str] = Field(
        default=None,
        description="Age range: 18-25, 26-35, 36-45, 46-55, 56+",
    )


class RegisterResponse(BaseModel):
    user_uid: str
    name: Optional[str] = None
    email: Optional[str] = None
    age_range: Optional[str] = None
    is_new: bool


class ProfileResponse(BaseModel):
    user_uid: str
    name: Optional[str] = None
    email: Optional[str] = None
    age_range: Optional[str] = None
    created_at: str


class UpdateProfileRequest(BaseModel):
    name: Optional[str] = None
    age_range: Optional[str] = None


# ---- Helpers ----

def _get_user(user_uid: str) -> Optional[dict]:
    eng = get_engine()
    with eng.begin() as conn:
        row = conn.execute(
            text(
                """
                SELECT user_id, email, name, age_range, created_at::text AS created_at
                FROM users WHERE user_id = :uid
                """
            ),
            {"uid": user_uid},
        ).fetchone()
    return dict(row._mapping) if row else None


def _create_or_update_user(
    user_uid: str,
    email: Optional[str],
    name: Optional[str],
    age_range: Optional[str],
) -> bool:
    """Insert or update user. Returns True if new user created."""
    eng = get_engine()
    with eng.begin() as conn:
        existing = conn.execute(
            text("SELECT 1 FROM users WHERE user_id = :uid"),
            {"uid": user_uid},
        ).fetchone()

        if existing:
            # Update fields if provided
            updates = []
            params = {"uid": user_uid}
            if email is not None:
                updates.append("email = :email")
                params["email"] = email
            if name is not None:
                updates.append("name = :name")
                params["name"] = name
            if age_range is not None:
                updates.append("age_range = :age_range")
                params["age_range"] = age_range
            updates.append("last_seen_at = now()")

            if updates:
                conn.execute(
                    text(f"UPDATE users SET {', '.join(updates)} WHERE user_id = :uid"),
                    params,
                )
            return False
        else:
            conn.execute(
                text(
                    """
                    INSERT INTO users (user_id, email, name, age_range)
                    VALUES (:uid, :email, :name, :age_range)
                    """
                ),
                {"uid": user_uid, "email": email, "name": name, "age_range": age_range},
            )
            return True


# ---- Endpoints ----

@router.post("/register", response_model=RegisterResponse)
def register(req: RegisterRequest) -> dict:
    """
    Register or login with Apple Sign In.

    First-time: creates user with profile info.
    Returning: updates last_seen, returns existing profile.
    """
    claims = verify_apple_token(req.apple_id_token)
    user_uid = claims["sub"]

    # Apple may provide email in the token
    email = req.email or claims.get("email")
    name = req.name

    is_new = _create_or_update_user(user_uid, email, name, req.age_range)

    user = _get_user(user_uid)
    logger.info(
        "auth %s: user_uid=%s email=%s",
        "register" if is_new else "login",
        user_uid[:12],
        email,
    )

    return {
        "user_uid": user_uid,
        "name": user.get("name") if user else name,
        "email": user.get("email") if user else email,
        "age_range": user.get("age_range") if user else req.age_range,
        "is_new": is_new,
    }


@router.get("/me", response_model=ProfileResponse)
def get_profile(
    x_apple_id_token: Optional[str] = Header(default=None),
    x_app_token: Optional[str] = Header(default=None),
    user_uid: Optional[str] = None,
) -> dict:
    """Get the authenticated user's profile."""
    # Verify identity
    if x_apple_id_token:
        claims = verify_apple_token(x_apple_id_token)
        verified_uid = claims["sub"]
    elif settings.app_token and x_app_token == settings.app_token and user_uid:
        verified_uid = user_uid  # legacy fallback
    else:
        raise HTTPException(status_code=401, detail="unauthorized")

    user = _get_user(verified_uid)
    if not user:
        raise HTTPException(status_code=404, detail="user_not_found")

    return {
        "user_uid": user["user_id"],
        "name": user.get("name"),
        "email": user.get("email"),
        "age_range": user.get("age_range"),
        "created_at": user["created_at"],
    }


@router.put("/me", response_model=ProfileResponse)
def update_profile(
    req: UpdateProfileRequest,
    x_apple_id_token: Optional[str] = Header(default=None),
    x_app_token: Optional[str] = Header(default=None),
    user_uid: Optional[str] = None,
) -> dict:
    """Update the authenticated user's profile."""
    if x_apple_id_token:
        claims = verify_apple_token(x_apple_id_token)
        verified_uid = claims["sub"]
    elif settings.app_token and x_app_token == settings.app_token and user_uid:
        verified_uid = user_uid
    else:
        raise HTTPException(status_code=401, detail="unauthorized")

    _create_or_update_user(verified_uid, None, req.name, req.age_range)
    user = _get_user(verified_uid)
    if not user:
        raise HTTPException(status_code=404, detail="user_not_found")

    return {
        "user_uid": user["user_id"],
        "name": user.get("name"),
        "email": user.get("email"),
        "age_range": user.get("age_range"),
        "created_at": user["created_at"],
    }
