import hashlib
import uuid
from datetime import datetime, timezone

import jwt
from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from neuralake.config.settings import get_settings
from neuralake.storage.database import get_db
from neuralake.storage.models.tenant import APIKey, Tenant, User

bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def _decode_jwt(token: str) -> dict:
    settings = get_settings()
    try:
        return jwt.decode(
            token,
            settings.auth.jwt_secret,
            algorithms=[settings.auth.jwt_algorithm],
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def create_jwt(user_id: uuid.UUID, tenant_id: uuid.UUID, role: str) -> str:
    settings = get_settings()
    from datetime import timedelta

    payload = {
        "sub": str(user_id),
        "tenant_id": str(tenant_id),
        "role": role,
        "exp": datetime.now(timezone.utc)
        + timedelta(hours=settings.auth.jwt_expiry_hours),
    }
    return jwt.encode(
        payload, settings.auth.jwt_secret, algorithm=settings.auth.jwt_algorithm
    )


async def _resolve_api_key(
    raw_key: str, db: AsyncSession
) -> tuple[uuid.UUID, list[str]]:
    settings = get_settings()
    if not raw_key.startswith(settings.auth.api_key_prefix):
        raise HTTPException(status_code=401, detail="Invalid API key format")

    key_hash = _hash_key(raw_key)
    result = await db.execute(
        select(APIKey).where(APIKey.key_hash == key_hash)
    )
    api_key = result.scalar_one_or_none()

    if api_key is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if api_key.expires_at and api_key.expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="API key expired")

    return api_key.tenant_id, api_key.scopes or []


class AuthContext:
    def __init__(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID | None = None,
        role: str = "api",
        scopes: list[str] | None = None,
    ):
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.role = role
        self.scopes = scopes or []


async def get_auth_context(
    request: Request,
    bearer: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
    api_key: str | None = Security(api_key_header),
    db: AsyncSession = Depends(get_db),
) -> AuthContext:
    if bearer and bearer.credentials:
        payload = _decode_jwt(bearer.credentials)
        return AuthContext(
            tenant_id=uuid.UUID(payload["tenant_id"]),
            user_id=uuid.UUID(payload["sub"]),
            role=payload.get("role", "user"),
        )

    if api_key:
        tenant_id, scopes = await _resolve_api_key(api_key, db)
        return AuthContext(tenant_id=tenant_id, scopes=scopes)

    raise HTTPException(
        status_code=401,
        detail="Provide a Bearer token or X-API-Key header",
    )


def require_role(*roles: str):
    async def _check(auth: AuthContext = Depends(get_auth_context)):
        if auth.role not in roles:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return auth

    return _check
