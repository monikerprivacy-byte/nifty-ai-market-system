"""Async Job Queue — API calls go through queue. Retry, backoff, no data loss."""

import asyncio, logging, uuid, json
from datetime import datetime
from collections import deque

logger = logging.getLogger("queue")

class Job:
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"

    def __init__(self, name, handler, args=None, kwargs=None, max_retries=3):
        self.id = uuid.uuid4().hex[:12]
        self.name = name
        self.handler = handler
        self.args = args or []
        self.kwargs = kwargs or {}
        self.max_retries = max_retries
        self.attempts = 0
        self.status = self.PENDING
        self.result = None
        self.error = None
        self.created_at = datetime.now()
        self.completed_at = None

class JobQueue:
    def __init__(self, max_concurrent=5):
        self._queue = deque()
        self._running = set()
        self._completed = []
        self._max_concurrent = max_concurrent
        self._max_history = 1000
        self._worker_task = None
        self._running_flag = False

    def enqueue(self, name, handler, args=None, kwargs=None, max_retries=3):
        job = Job(name, handler, args, kwargs, max_retries)
        self._queue.append(job)
        logger.info(f"Queued job {job.id}: {name}")
        return job

    async def start(self):
        self._running_flag = True
        self._worker_task = asyncio.create_task(self._worker())
        logger.info("Job queue worker started")

    async def stop(self):
        self._running_flag = False
        if self._worker_task:
            self._worker_task.cancel()
        logger.info("Job queue worker stopped")

    async def _worker(self):
        while self._running_flag:
            if len(self._running) < self._max_concurrent and self._queue:
                job = self._queue.popleft()
                self._running.add(job)
                asyncio.create_task(self._execute(job))
            else:
                await asyncio.sleep(0.1)

    async def _execute(self, job):
        job.status = Job.RUNNING
        while job.attempts <= job.max_retries:
            job.attempts += 1
            try:
                if asyncio.iscoroutinefunction(job.handler):
                    job.result = await job.handler(*job.args, **job.kwargs)
                else:
                    job.result = job.handler(*job.args, **job.kwargs)
                job.status = Job.SUCCESS
                job.completed_at = datetime.now()
                logger.info(f"Job {job.id} ({job.name}) completed")
                break
            except Exception as e:
                job.error = str(e)
                if job.attempts <= job.max_retries:
                    job.status = Job.RETRYING
                    wait = min(5 * (2 ** (job.attempts - 1)), 60)
                    logger.warning(f"Job {job.id} failed (attempt {job.attempts}/{job.max_retries}): {e}. Retry in {wait}s")
                    await asyncio.sleep(wait)
                else:
                    job.status = Job.FAILED
                    job.completed_at = datetime.now()
                    logger.error(f"Job {job.id} failed after {job.attempts} attempts: {e}")

        self._running.discard(job)
        self._completed.append(job)
        if len(self._completed) > self._max_history:
            self._completed.pop(0)

    def status(self, job_id=None):
        if job_id:
            for j in list(self._queue) + list(self._running) + self._completed:
                if j.id == job_id:
                    return {"id": j.id, "name": j.name, "status": j.status,
                            "attempts": j.attempts, "error": j.error}
            return None
        return {
            "queued": len(self._queue),
            "running": len(self._running),
            "completed": len([j for j in self._completed if j.status == Job.SUCCESS]),
            "failed": len([j for j in self._completed if j.status == Job.FAILED]),
        }

# Global singleton
_queue = None

def get_queue():
    global _queue
    if _queue is None:
        _queue = JobQueue()
    return _queue
