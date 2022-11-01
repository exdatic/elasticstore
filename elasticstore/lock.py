import asyncio
import logging
import uuid

from elasticsearch import AsyncElasticsearch

ACQUIRE_SCRIPT = """
    def now = System.currentTimeMillis();
    if (ctx.op == 'create' || now > ctx._source.lock_until) {
        ctx._source.locked_by = params.owner;
        ctx._source.locked_at = now;
        ctx._source.lock_until = now + params.duration * 1000;
    } else {
        ctx.op = 'none';
    }
"""

REFRESH_SCRIPT = """
    def now = System.currentTimeMillis();
    if (params.owner == ctx._source.locked_by) {
        ctx._source.locked_by = params.owner;
        ctx._source.locked_at = now;
        ctx._source.lock_until = now + params.duration * 1000;
    } else {
        ctx.op = 'none';
    }
"""

RELEASE_SCRIPT = """
    def now = System.currentTimeMillis();
    if (params.owner == ctx._source.locked_by) {
        ctx._source.lock_until = now;
    } else {
        ctx.op = 'none';
    }
"""

DEFAULT_RETRY = 100


class AlreadyLockedError(Exception):
    pass


class Lock():

    def __init__(
        self,
        es: AsyncElasticsearch,
        resource: str,
        duration: int = 120,
        refresh_interval: int = 90,
        index: str = "locks"
    ):
        """
        Distributed lock based on Elasticsearch

        Parameters
        ----------
        es : AsyncElasticsearch
            The Elasticsearch client
        resource : str
            The locked resource
        duration : int
            Period in seconds, which indicates how long the resource is blocked
        refresh_interval : int
            Period in seconds after which the lock is refreshed
        index : str
            The Elasticsearch index name
        """
        self._es = es
        self._index = index
        self._resource = resource
        self._duration = duration
        self._refresh_interval = refresh_interval
        self._refresh_task = None
        self._owner = uuid.uuid4().hex

    @property
    def owner(self):
        return self._owner

    async def _update(self, **kwargs):
        return await self._es.update(
            index=self._index,
            id=self._resource,
            body=kwargs,
            retry_on_conflict=DEFAULT_RETRY,
            refresh="wait_for")

    async def _acquire_lock(self):
        # to run the script whether or not the document exists,
        # set scripted_upsert to true
        res = await self._update(
            scripted_upsert=True,
            script=dict(
                source=ACQUIRE_SCRIPT,
                params=dict(owner=self._owner, duration=self._duration)),
            upsert={})
        return res["result"] in ("created", "updated")

    async def _refresh_lock(self):
        res = await self._update(
            script=dict(
                source=REFRESH_SCRIPT,
                params=dict(owner=self._owner, duration=self._duration)))
        return res["result"] in ("updated")

    async def _release_lock(self):
        res = await self._update(
            script=dict(
                source=RELEASE_SCRIPT,
                params=dict(owner=self.owner)))
        return res["result"] in ("updated")

    async def _refresh_loop(self):
        while True:
            await asyncio.sleep(self._refresh_interval)
            if not await self._refresh_lock():
                logging.warning(
                    f'Lock for "{self._resource}" could not be refreshed')
                break

    async def __aenter__(self):
        if not await self._acquire_lock():
            raise AlreadyLockedError(
                f'Lock for "{self._resource}" could not be acquired')
        self._refresh_task = asyncio.create_task(self._refresh_loop())
        return self

    async def __aexit__(self, exc_type, exc, tb):
        assert self._refresh_task is not None
        self._refresh_task.cancel()
        if not await self._release_lock():
            logging.warning(
                f'Lock for "{self._resource}" could not be released')
