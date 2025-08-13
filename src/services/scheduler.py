"""
Simple APScheduler-based job scheduler for CI detectors.
"""

from __future__ import annotations

from typing import Optional
import logging
import asyncio

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from .change_detection_service import change_detection_service
from .news_ingestion_service import news_ingestion_service

logger = logging.getLogger(__name__)


class CIScheduler:
    def __init__(self) -> None:
        self.scheduler: Optional[AsyncIOScheduler] = None

    async def start(self) -> None:
        if self.scheduler:
            return
        self.scheduler = AsyncIOScheduler()

        # Pricing detection every 4 hours
        self.scheduler.add_job(
            lambda: asyncio.create_task(change_detection_service.detect_pricing_for_all()),
            trigger=CronTrigger(minute=0, hour='*/4'),
            id='ci_pricing_every_4h',
            replace_existing=True,
        )

        # Copy/features detection every 6 hours
        self.scheduler.add_job(
            lambda: asyncio.create_task(change_detection_service.detect_copy_for_all()),
            trigger=CronTrigger(minute=15, hour='*/6'),
            id='ci_copy_every_6h',
            replace_existing=True,
        )

        # News ingestion every 3 hours
        self.scheduler.add_job(
            lambda: asyncio.create_task(news_ingestion_service.ingest_for_all(days_back=2)),
            trigger=CronTrigger(minute=30, hour='*/3'),
            id='ci_news_every_3h',
            replace_existing=True,
        )

        self.scheduler.start()
        logger.info("CIScheduler started with pricing, copy and news jobs")

    async def shutdown(self) -> None:
        if self.scheduler:
            self.scheduler.shutdown(wait=False)
            self.scheduler = None
            logger.info("CIScheduler stopped")


ci_scheduler = CIScheduler()


