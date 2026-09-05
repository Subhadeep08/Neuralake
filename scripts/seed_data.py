"""Seed script to create a default tenant, user, and API key for development."""

import asyncio
import hashlib
import secrets

from neuralake.config.settings import get_settings
from neuralake.core.auth.password import hash_password
from neuralake.storage.database import get_session_factory, init_db
from neuralake.storage.models.tenant import APIKey, Tenant, User


async def seed():
    await init_db()
    factory = get_session_factory()
    settings = get_settings()

    async with factory() as db:
        tenant = Tenant(
            name="Default Tenant",
            slug="default",
            plan="free",
            settings={},
        )
        db.add(tenant)
        await db.flush()

        user = User(
            tenant_id=tenant.id,
            email="admin@neuralake.dev",
            name="Admin",
            password_hash=hash_password("admin123"),
            role="admin",
        )
        db.add(user)

        raw_key = settings.auth.api_key_prefix + secrets.token_urlsafe(32)
        api_key = APIKey(
            tenant_id=tenant.id,
            key_hash=hashlib.sha256(raw_key.encode()).hexdigest(),
            key_prefix=raw_key[:12],
            scopes=["read", "write", "admin"],
        )
        db.add(api_key)
        await db.commit()

        print(f"Tenant created: {tenant.name} (id: {tenant.id})")
        print(f"User created: {user.email}")
        print(f"API Key: {raw_key}")


if __name__ == "__main__":
    asyncio.run(seed())
