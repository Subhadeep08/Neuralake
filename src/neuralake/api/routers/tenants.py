import secrets
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from neuralake.api.middleware.auth import AuthContext, create_jwt, get_auth_context
from neuralake.core.auth.password import hash_password, needs_rehash, verify_password
from neuralake.api.schemas.requests import (
    CreateAPIKeyRequest,
    CreateTenantRequest,
    CreateUserRequest,
    LoginRequest,
)
from neuralake.api.schemas.responses import (
    APIKeyResponse,
    TenantResponse,
    TokenResponse,
    UserResponse,
)
from neuralake.config.settings import get_settings
from neuralake.storage.database import get_db
from neuralake.storage.models.tenant import APIKey, Tenant, User

router = APIRouter(prefix="/tenants", tags=["tenants"])


@router.post("", response_model=TenantResponse, status_code=201)
async def create_tenant(
    body: CreateTenantRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    settings = get_settings()
    has_tenants = (await db.execute(select(Tenant).limit(1))).scalar_one_or_none()

    if has_tenants:
        token = request.headers.get("X-Bootstrap-Token")
        if not settings.auth.bootstrap_token or token != settings.auth.bootstrap_token:
            raise HTTPException(status_code=403, detail="Bootstrap token required")

    tenant = Tenant(
        name=body.name,
        slug=body.slug,
        plan=body.plan,
        settings=body.settings or {},
    )
    db.add(tenant)
    await db.flush()
    return TenantResponse.model_validate(tenant)


@router.post("/{tenant_id}/users", response_model=UserResponse, status_code=201)
async def create_user(
    tenant_id: uuid.UUID,
    body: CreateUserRequest,
    db: AsyncSession = Depends(get_db),
):
    user = User(
        tenant_id=tenant_id,
        email=body.email,
        name=body.name,
        password_hash=hash_password(body.password),
        role=body.role,
    )
    db.add(user)
    await db.flush()
    return UserResponse.model_validate(user)


@router.post("/auth/login", response_model=TokenResponse)
async def login(
    body: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if needs_rehash(user.password_hash):
        user.password_hash = hash_password(body.password)
        await db.flush()

    token = create_jwt(user.id, user.tenant_id, user.role)
    return TokenResponse(access_token=token)


@router.post("/me/api-keys", response_model=APIKeyResponse, status_code=201)
async def create_api_key(
    body: CreateAPIKeyRequest,
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    settings = get_settings()
    raw_key = settings.auth.api_key_prefix + secrets.token_urlsafe(32)
    import hashlib

    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    expires_at = None
    if body.expires_in_days:
        expires_at = datetime.now(timezone.utc) + timedelta(days=body.expires_in_days)

    api_key = APIKey(
        tenant_id=auth.tenant_id,
        key_hash=key_hash,
        key_prefix=raw_key[:12],
        scopes=body.scopes,
        expires_at=expires_at,
    )
    db.add(api_key)
    await db.flush()

    resp = APIKeyResponse.model_validate(api_key)
    resp.raw_key = raw_key
    return resp


@router.get("/me/api-keys", response_model=list[APIKeyResponse])
async def list_api_keys(
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(APIKey).where(APIKey.tenant_id == auth.tenant_id)
    )
    keys = result.scalars().all()
    return [APIKeyResponse.model_validate(k) for k in keys]


@router.delete("/me/api-keys/{key_id}", status_code=204)
async def delete_api_key(
    key_id: uuid.UUID,
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(APIKey).where(
            APIKey.id == key_id, APIKey.tenant_id == auth.tenant_id
        )
    )
    api_key = result.scalar_one_or_none()
    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")
    await db.delete(api_key)
