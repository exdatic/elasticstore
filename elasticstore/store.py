from datetime import datetime
import functools
from typing import (
    Any, AsyncGenerator, AsyncIterable, Dict, Generic, Iterable, List,
    Optional, Tuple, Type, TypeVar, Union, cast
)
from typing_extensions import Literal

from elasticsearch import AsyncElasticsearch, NotFoundError, RequestError
from elasticsearch.helpers import async_scan, async_bulk
from elasticsearch_dsl import Document
from elasticsearch_dsl.response import Hit
from loguru import logger
import orjson
from pydantic import BaseModel

from elasticstore.utils import aiter, chunked


DEFAULT_SETTINGS = {
    "index": {
        "number_of_shards": 5,
        "number_of_replicas": 1
    }
}

DEFAULT_RETRY = 100

# params must be a dict of objects
UPDATE_SCRIPT = "params.forEach((k, v) -> ctx._source[k] = v)"

# params must be a dict of lists
APPEND_TO_LISTS_SCRIPT = """
    params.forEach((k, v) -> {
        List oldValues = ctx._source[k];
        if (oldValues != null) {
            Set newValues = new LinkedHashSet(oldValues);
            newValues.addAll(v);
            ctx._source[k] = new ArrayList(newValues);
        } else {
            ctx._source[k] = v;
        }
    })
"""

# params must be a dict of lists
REMOVE_FROM_LISTS_SCRIPT = """
    params.forEach((k, v) -> {
        List oldValues = ctx._source[k];
        if (oldValues != null) {
            Set values = new HashSet(v);
            oldValues.removeIf(item -> values.contains(item));
            ctx._source[k] = oldValues;
        }
        return null;
    })
"""

T = TypeVar('T', bound=Union[Dict[str, Any], BaseModel])


def ensure_index_exists():
    def wrapper(func):
        @functools.wraps(func)
        async def wrapped(*args, **kwargs):
            items = cast(Store, args[0])
            await items.index_create()
            return await func(*args, **kwargs)
        return wrapped
    return wrapper


def mapping_dict(mappings: Union[Dict[str, Any], Type[Document]]):
    if mappings and isinstance(mappings, Type) and issubclass(mappings, Document):
        return cast(Dict[str, Any], getattr(mappings, '_doc_type').mapping.to_dict())
    return mappings


class Store(Generic[T]):

    def __init__(
        self,
        es: AsyncElasticsearch,
        index: str,
        model_type: Type[T] = dict,
        mappings: Optional[Union[Dict[str, Any], Type[Document]]] = None,
        settings: Optional[Dict] = DEFAULT_SETTINGS
    ):
        self._es = es
        self._index = index
        self._model_type = model_type
        self._mappings = mappings
        self._settings = settings

    def _to_dict(self, item: Union[Dict[str, Any], BaseModel]) -> Dict:
        if isinstance(item, BaseModel):
            return orjson.loads(item.json())
        else:
            return item

    def _from_dict(self, doc: Dict) -> T:
        return self._model_type(**doc)

    @property
    def index(self):
        """
        Returns index name.
        """
        return self._index

    async def index_exists(self):
        """
        Checks whether the index exists.
        """
        return await self._es.indices.exists(index=self._index)

    async def index_create(self):
        """
        Creates the index if not exists.
        """
        if not await self.index_exists():
            body = dict()
            if self._mappings:
                body["mappings"] = mapping_dict(self._mappings)
            if self._settings:
                body["settings"] = self._settings
            return await self._es.indices.create(index=self._index, body=body)

    async def index_update(self):
        """
        Update index mappings.
        """
        if not await self.index_exists():
            await self.index_create()
        else:
            if self._mappings:
                try:
                    body = mapping_dict(self._mappings)
                    await self._es.indices.put_mapping(index=self._index, body=body)
                except RequestError as e:
                    logger.warning(f"Could not update mappings of index '{self._index}' ({e})")

    async def index_rebuild(self):
        """
        Rebuild index (should fix mapping errors).
        """
        next_index = self._index + '-' + datetime.now().strftime("%Y%m%d%H%M%S%f")

        body = dict()
        if self._mappings:
            body["mappings"] = mapping_dict(self._mappings)
        if self._settings:
            body["settings"] = self._settings

        # create new index
        await self._es.indices.create(index=next_index, body=body)

        # move data
        await self._es.reindex(
            body={"source": {"index": self._index}, "dest": {"index": next_index}},
            # request_timeout=3600
        )

        # refresh the index to make the changes visible
        await self._es.indices.refresh(index=next_index)

        # a) move alias (without downtime)
        aliases = await self._es.indices.get_alias(index=self._index, ignore_unavailable=True)
        if aliases and self._index not in aliases:
            alias = self._index
            index = next(iter(aliases.keys()))
            await self._es.indices.update_aliases(body={
                "actions": [
                    {"remove": {"alias": alias, "index": index}},
                    {"add": {"alias": alias, "index": next_index}},
                ]
            })
            await self._es.indices.delete(index=index)
        # b) delete old index and set index alias on new index (with downtime)
        else:
            await self._es.indices.delete(index=self._index)
            await self._es.indices.update_aliases(body={
                "actions": [
                    {"add": {"alias": self._index, "index": next_index}},
                ]
            })

    async def index_copy(self, index_name: str):
        """
        Rebuild index (should fix mapping errors).
        """
        body = dict()
        if self._mappings:
            body["mappings"] = mapping_dict(self._mappings)
        if self._settings:
            body["settings"] = self._settings

        # create new index
        await self._es.indices.create(index=index_name, body=body)

        # move data
        await self._es.reindex(
            body={"source": {"index": self._index}, "dest": {"index": index_name}},
            # request_timeout=3600
        )

        # refresh the index to make the changes visible
        await self._es.indices.refresh(index=index_name)

    async def index_delete(self):
        """
        Deletes the index.
        """
        try:
            return await self._es.indices.delete(index=self._index)
        except NotFoundError:
            return None

    async def has(self, id: str) -> bool:
        """
        Checks whether the document exists in the index.
        """
        try:
            return await self._es.exists(index=self._index, id=id)
        except NotFoundError:
            return False

    async def get(self, id: str, **kwargs) -> T:
        """
        Returns doc.
        """
        try:
            res = await self._es.get(index=self._index, id=id, **kwargs)
        except NotFoundError:
            raise KeyError(id) from None

        if not res["found"]:
            raise KeyError(id)

        hit = Hit(res)
        doc = cast(Dict, hit.to_dict())
        return self._from_dict(doc)

    async def delete(
        self, id: str, refresh: Union[bool, Literal['wait_for']] = False
    ):
        """
        Removes a document from the index.
        """
        try:
            return await self._es.delete(index=self._index, id=id, refresh=refresh)
        except NotFoundError:
            raise KeyError(id) from None

    @ensure_index_exists()
    async def update(
        self, id: str, item: T, refresh: Union[bool, Literal['wait_for']] = False
    ):
        """
        Creates or updates a document in the index.
        """
        doc = self._to_dict(item)
        return await self._es.index(
            index=self._index, body=doc, id=id, refresh=refresh)  # type: ignore (FIXME es8)

    @ensure_index_exists()
    async def upsert(
        self,
        id: str,
        item: Union[Dict[str, Any], BaseModel],
        source: Optional[str] = None,
        create: bool = False,
        retry: int = DEFAULT_RETRY,
        refresh: Union[bool, Literal['wait_for']] = False
    ):
        """
        Updates a document in the index.
        Interprets items as script params if a script source is given.
        """
        try:
            doc = self._to_dict(item)
            if source is not None:
                script = dict(source=source, params=doc)
                if create:
                    op = dict(script=script, scripted_upsert=True, upsert={})
                else:
                    op = dict(script=script)
            else:
                if create:
                    op = dict(doc=doc, doc_as_upsert=True)
                else:
                    op = dict(doc=doc)

            return await self._es.update(
                index=self._index, id=id, body=op, retry_on_conflict=retry, refresh=refresh)

        except NotFoundError:
            raise KeyError(id) from None

    @ensure_index_exists()
    async def bulk_delete(
        self,
        ids: Union[Iterable[str], AsyncIterable[str]]
    ):
        """
        Remove documents from the index.
        """
        async def iter_actions():
            async for id in aiter(ids):
                yield dict(_index=self._index, _id=id, _op_type='delete')

        return await async_bulk(self._es, iter_actions(), raise_on_error=False)

    @ensure_index_exists()
    async def bulk_update(
        self,
        items: AsyncIterable[Tuple[str, T]],
        retry: int = DEFAULT_RETRY
    ):
        """
        Updates documents in the index.
        """
        async def iter_actions():
            async for key, item in items:
                doc = self._to_dict(item)
                # use _source=doc and not **doc as it allows fields with reserved names like "version"
                yield dict(_index=self._index, _id=key, _op_type='index', retry_on_conflict=retry, _source=doc)

        return await async_bulk(self._es, iter_actions(), raise_on_error=False)

    @ensure_index_exists()
    async def bulk_upsert(
        self,
        items: AsyncIterable[Tuple[str, Union[Dict[str, Any], BaseModel]]],
        source: Optional[str] = None,
        create: bool = False,
        retry: int = DEFAULT_RETRY
    ):
        """
        Updates documents in the index with partial content.
        Interprets items as script params if a script source is given.
        """
        async def iter_actions():
            async for key, item in items:
                doc = self._to_dict(item)
                if source is not None:
                    script = dict(source=source, params=doc)
                    if create:
                        # To run the script whether or not the document exists,
                        # set scripted_upsert to true.
                        yield dict(
                            _index=self._index, _id=key, _op_type='update', retry_on_conflict=retry,
                            script=script, scripted_upsert=True, upsert={})
                    else:
                        yield dict(
                            _index=self._index, _id=key, _op_type='update', retry_on_conflict=retry,
                            script=script)
                else:
                    # Instead of sending a partial doc plus an upsert doc,
                    # you can set doc_as_upsert to true to use the contents
                    # of doc as the upsert value.
                    if create:
                        yield dict(
                            _index=self._index, _id=key, _op_type='update', retry_on_conflict=retry,
                            doc=doc, doc_as_upsert=True)
                    else:
                        yield dict(
                            _index=self._index, _id=key, _op_type='update', retry_on_conflict=retry,
                            doc=doc)

        return await async_bulk(self._es, iter_actions(), raise_on_error=False)

    @ensure_index_exists()
    async def delete_by_query(self, query: Dict):
        """
        Deletes documents matching the provided query.
        """
        return await self._es.delete_by_query(
            index=self._index,
            body=query,
            params=dict(conflicts='proceed'))

    @ensure_index_exists()
    async def update_by_query(
        self,
        query: Dict,
        item: Union[Dict[str, Any], BaseModel],
        source: Optional[str]
    ):
        """
        Update documents matching the provided query.
        Interprets items as script params.
        """
        doc = self._to_dict(item)
        return await self._es.update_by_query(
            index=self._index,
            body=dict(script=dict(source=source, params=doc), **query),
            params=dict(conflicts='proceed'))

    async def mget(
        self, ids: Union[Iterable[str], AsyncIterable[str]], chunk_size: int = 500
    ) -> AsyncGenerator[Tuple[str, Optional[T]], None]:
        async for chunk in chunked(ids, chunk_size):
            res = await self._es.mget(index=self._index, body={'ids': chunk}, ignore=(404))
            if "docs" in res:
                for res in res["docs"]:
                    if res.get("found", False):
                        doc = cast(Dict, Hit(res).to_dict())
                        yield res["_id"], self._from_dict(doc)
                    else:
                        yield res["_id"], None

    async def refresh(self):
        await self._es.indices.refresh(index=self._index)

    async def flush(self):
        await self._es.indices.flush(index=self._index)

    def __aiter__(self):
        return self.keys()

    async def keys(self):
        try:
            async for res in async_scan(self._es, index=self._index, _source=False):
                assert isinstance(res, dict)
                yield res["_id"]
        except NotFoundError:
            pass

    async def values(self):
        async for _, item in self.items():
            yield item

    async def items(self, query: Optional[Dict] = None, **kwargs) -> AsyncGenerator[Tuple[str, T], None]:
        """
        Return iterator of docs.

        A query may look like this:
        ```json
        {
            "query": {
                "match_all": {}
            }
        }
        ```
        """
        try:
            async for res in async_scan(self._es, index=self._index, query=query, **kwargs):
                assert isinstance(res, dict)
                doc = cast(Dict, Hit(res).to_dict())
                yield res["_id"], self._from_dict(doc)
        except NotFoundError:
            pass

    async def count(self, query: Optional[Dict] = None) -> int:
        try:
            res = await self._es.count(index=self._index, body=query)
            return res['count']
        except NotFoundError:
            return 0

    async def search(self, query: Dict, **kwargs):
        try:
            return await self._es.search(index=self._index, body=query, **kwargs)
        except NotFoundError:
            return None

    async def filter_by(
        self,
        fields: Dict[str, Union[Union[str, int, float, bool], List[Union[str, int, float, bool]]]]
    ) -> List[T]:
        return [item async for _, item in self.items(query={
            "query": {
                "bool": {
                    "filter": [
                        {
                            "terms": {
                                (field): values if isinstance(values, (list, tuple)) else [values]
                            }
                        } for field, values in fields.items()
                    ]
                }
            }
        })]
