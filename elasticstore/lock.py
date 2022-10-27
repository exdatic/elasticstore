import asyncio
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

RELEASE_SCRIPT = """
    def now = System.currentTimeMillis();
    if (params.owner == ctx._source.locked_by) {
        ctx._source.lock_until = now;
    } else {
        ctx.op = 'none';
    }
"""


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
        index : str, optional
            The Elasticsearch index name
        resource : str
            The locked resource
        duration : int, optional
            Period in seconds, which indicates how long the resource is blocked
        refresh_interval : int, optional
            Period in seconds after which the lock is refreshed
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

    async def _acquire(self):
        script = dict(source=ACQUIRE_SCRIPT, params=dict(owner=self._owner, duration=self._duration))
        req = dict(scripted_upsert=True, script=script, upsert={})
        res = await self._es.update(index=self._index, id=self._resource, body=req, refresh=True)
        return res["result"] in ("created", "updated")

    async def _release(self):
        script = dict(source=RELEASE_SCRIPT, params=dict(owner=self.owner))
        req = dict(script=script)
        res = await self._es.update(index=self._index, id=self._resource, body=req, refresh=True)
        return res["result"] in ("updated")

    async def _refresh(self):
        while True:
            await asyncio.sleep(self._refresh_interval)
            if not await self._acquire():
                break

    async def __aenter__(self):
        if not await self._acquire():
            raise AlreadyLockedError(f'Lock for "{self._resource}" could not be acquired by "{self._owner}"')
        self._refresh_task = asyncio.create_task(self._refresh())
        return self

    async def __aexit__(self, exc_type, exc, tb):
        assert self._refresh_task is not None
        self._refresh_task.cancel()
        if not await self._release():
            raise RuntimeError(f'Lock for "{self._resource}" could not be released by "{self._owner}"')
