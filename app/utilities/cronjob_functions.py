import logging.config

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from app.core.secrets import config_secrets
from apscheduler.triggers.cron import CronTrigger



class Cronjob:
    scheduler = AsyncIOScheduler()

    async def dump_qdrant_data():

        url = f"{config_secrets.APP_BACKEND_URL}api/v1/qdrant/dump-data"
        async with httpx.AsyncClient() as client:
            await client.post(url)
            return True


        Cronjob.scheduler.start()

    async def start_scheduler():
        Cronjob.scheduler.add_job(
            Cronjob.dump_qdrant_data,
            CronTrigger(hour=0, minute=0),
            id="dump_qdrant_data",
            coalesce=True,
            replace_existing=True,
        )


        Cronjob.scheduler.start()