import asyncio
import shutil

async def delayed_cleanup(path: str, delay: int = 60):
    await asyncio.sleep(delay)
    shutil.rmtree(path, ignore_errors=True)
