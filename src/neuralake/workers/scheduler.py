import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from neuralake.config.settings import get_settings

logger = logging.getLogger(__name__)

_scheduler: AsyncIOScheduler | None = None


async def start_scheduler():
    global _scheduler
    settings = get_settings()

    if not settings.memory.consolidation_enabled:
        logger.info("Memory consolidation disabled")
        return

    _scheduler = AsyncIOScheduler()

    _scheduler.add_job(
        _run_consolidation,
        trigger=CronTrigger.from_crontab(settings.memory.consolidation_cron),
        id="memory_consolidation",
        name="Memory Consolidation",
        replace_existing=True,
    )

    _scheduler.start()
    logger.info("Scheduler started with consolidation cron: %s", settings.memory.consolidation_cron)


async def stop_scheduler():
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None


async def _run_consolidation():
    from sqlalchemy import select

    from neuralake.core.memory.consolidator import MemoryConsolidator
    from neuralake.storage.database import get_session_factory
    from neuralake.storage.models.tenant import Tenant

    logger.info("Starting memory consolidation job")
    factory = get_session_factory()

    async with factory() as db:
        result = await db.execute(select(Tenant))
        tenants = result.scalars().all()

        consolidator = MemoryConsolidator()
        for tenant in tenants:
            try:
                stats = await consolidator.consolidate(tenant_id=tenant.id, db=db)
                logger.info(
                    "Tenant %s consolidation: %s", tenant.slug, stats
                )
            except Exception as e:
                logger.error("Consolidation failed for %s: %s", tenant.slug, e)

        await db.commit()
