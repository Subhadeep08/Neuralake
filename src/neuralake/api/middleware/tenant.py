from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from neuralake.api.middleware.auth import AuthContext, get_auth_context
from neuralake.storage.database import get_db, set_tenant_context


async def set_tenant_rls(
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> AuthContext:
    await set_tenant_context(db, str(auth.tenant_id))
    return auth
