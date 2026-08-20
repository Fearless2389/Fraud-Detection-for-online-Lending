"""API authentication.

A fraud scoring endpoint is a sensitive surface twice over: it returns a risk
assessment of a named individual, and an attacker who can query it freely can
probe the model's decision boundary to learn what gets through. Both are
reasons this API is never open.

Authentication here is a shared API key presented in the ``X-API-Key`` header.
For a prototype this is the right level: it is genuinely enforced on every
scoring route rather than mocked, while not pretending to be the full OAuth /
mTLS arrangement a production lending platform would use. The upgrade path is
documented in the README rather than half-built here.
"""

from __future__ import annotations

import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

from app.core.config import Settings, get_settings

# ``auto_error=False`` so a missing header produces our own 401 with a useful
# message, rather than FastAPI's default 403.
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(
    provided_key: str | None = Depends(_api_key_header),
    settings: Settings = Depends(get_settings),
) -> str:
    """Reject any request without a valid API key.

    The comparison uses :func:`secrets.compare_digest` rather than ``==``. A
    plain equality check short-circuits on the first differing byte, and the
    timing difference is measurable across enough requests, which lets an
    attacker recover the key one byte at a time. Constant-time comparison
    costs nothing and removes the class of attack entirely.
    """
    if provided_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header.",
            headers={"WWW-Authenticate": "APIKey"},
        )

    if not secrets.compare_digest(provided_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "APIKey"},
        )

    return provided_key
