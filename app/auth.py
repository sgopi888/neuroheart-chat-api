"""
Apple Sign In token verification.

Verifies Apple id_token (JWT) using Apple's public keys.
Extracts user_uid (sub claim) — this is the trusted identity.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import httpx
import jwt
from fastapi import Header, HTTPException

from app.config import settings

logger = logging.getLogger(__name__)

# Apple's public keys endpoint
_APPLE_KEYS_URL = "https://appleid.apple.com/auth/keys"
_APPLE_ISSUER = "https://appleid.apple.com"

# Cache Apple's public keys (refresh every 24h)
_cached_keys: list = []
_keys_fetched_at: float = 0
from app.config import settings as _cfg
_KEYS_TTL = _cfg.auth_apple_keys_ttl


def _fetch_apple_keys() -> list:
    """Fetch Apple's public signing keys (JWKS)."""
    global _cached_keys, _keys_fetched_at

    now = time.time()
    if _cached_keys and (now - _keys_fetched_at) < _KEYS_TTL:
        return _cached_keys

    try:
        resp = httpx.get(_APPLE_KEYS_URL, timeout=10)
        resp.raise_for_status()
        _cached_keys = resp.json().get("keys", [])
        _keys_fetched_at = now
        logger.info("Fetched %d Apple public keys", len(_cached_keys))
    except Exception as exc:
        logger.warning("Failed to fetch Apple keys: %s", exc)
        if _cached_keys:
            return _cached_keys
        raise

    return _cached_keys


def verify_apple_token(id_token: str) -> dict:
    """
    Verify an Apple id_token and return the decoded claims.

    Returns dict with at least:
      - sub: user's Apple ID (= user_uid)
      - email: user's email (if shared)
      - email_verified: bool
    """
    # Decode header to find the key ID
    try:
        unverified_header = jwt.get_unverified_header(id_token)
    except jwt.DecodeError as exc:
        raise HTTPException(status_code=401, detail=f"invalid_token: {exc}")

    kid = unverified_header.get("kid")
    if not kid:
        raise HTTPException(status_code=401, detail="token_missing_kid")

    # Find matching Apple public key
    apple_keys = _fetch_apple_keys()
    matching_key = None
    for key in apple_keys:
        if key.get("kid") == kid:
            matching_key = key
            break

    if not matching_key:
        # Keys may have rotated — force refresh
        _cached_keys.clear()
        apple_keys = _fetch_apple_keys()
        for key in apple_keys:
            if key.get("kid") == kid:
                matching_key = key
                break

    if not matching_key:
        raise HTTPException(status_code=401, detail="apple_key_not_found")

    # Build public key from JWK
    try:
        public_key = jwt.algorithms.RSAAlgorithm.from_jwk(matching_key)
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"key_parse_error: {exc}")

    # Verify and decode the token
    try:
        claims = jwt.decode(
            id_token,
            public_key,
            algorithms=["RS256"],
            issuer=_APPLE_ISSUER,
            audience=settings.apple_bundle_id,
            options={"verify_exp": True},
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="token_expired")
    except jwt.InvalidAudienceError:
        raise HTTPException(status_code=401, detail="invalid_audience")
    except jwt.InvalidIssuerError:
        raise HTTPException(status_code=401, detail="invalid_issuer")
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail=f"invalid_token: {exc}")

    return claims


def get_verified_user_uid(
    x_apple_id_token: Optional[str] = Header(default=None),
    x_app_token: Optional[str] = Header(default=None),
) -> str:
    """
    FastAPI dependency: extract verified user_uid.

    Priority:
    1. Apple id_token → verify with Apple → extract 'sub' claim
    2. Fallback to app_token + client-asserted user_uid (legacy, for dev/testing)
    """
    if x_apple_id_token:
        claims = verify_apple_token(x_apple_id_token)
        return claims["sub"]

    # Legacy fallback: app_token auth (no user verification)
    if settings.app_token and x_app_token == settings.app_token:
        return ""  # caller must provide user_uid in body/query

    raise HTTPException(status_code=401, detail="unauthorized")
