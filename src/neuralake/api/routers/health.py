from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from neuralake.api.schemas.responses import HealthResponse
from neuralake.storage.database import get_db

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health_check():
    from neuralake import __version__

    return HealthResponse(status="ok", version=__version__, database="unknown")


@router.get("/ready", response_model=HealthResponse)
async def readiness_check(db: AsyncSession = Depends(get_db)):
    from neuralake import __version__

    try:
        await db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {e}"

    status = "ok" if db_status == "connected" else "degraded"
    return HealthResponse(
        status=status,
        version=__version__,
        database=db_status,
    )
