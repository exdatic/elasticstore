from datetime import datetime
import functools
import logging
from typing import (
    Any, Generator, Iterable, Dict, Generic, List, Literal,
    Optional, Tuple, Type, TypeVar, Union, cast
)

from elasticsearch import Elasticsearch, NotFoundError, RequestError
from elasticsearch.helpers import scan, bulk
from elasticsearch_dsl import Document
from elasticsearch_dsl.response import Hit
import orjson

from elasticstore.utils import chunked


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

T = TypeVar('T', bound=Union[Dict[str, Any], Any])


def ensure_index_exists():
    def wrapper(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            items = cast(Store, args[0])
            items.index_create()
            return func(*args, **kwargs)
        return wrapped
    return wrapper


def mappings_to_dict(mappings: Union[Dict[str, Any], Type[Document]]):
    if mappings and isinstance(mappings, Type) and issubclass(mappings, Document):
        return cast(Dict[str, Any], getattr(mappings, '_doc_type').mapping.to_dict())
    return mappings


class Store(Generic[T]):

    def __init__(
        self,
        es: Elasticsearch,
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

    def _to_dict(self, item: Union[Dict[str, Any], Any]) -> Dict:
        # Pydantic V1
        if hasattr(item, "json"):
            return orjson.loads(getattr(item, "json")())
        # Pydantic V2
        if hasattr(item, "model_dump_json"):
            return orjson.loads(getattr(item, "json")())
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

    def index_exists(self):
        """
        Checks whether the index exists.
        """
        resp = self._es.indices.exists(index=self._index)
        return resp.body

    def index_create(self):
        """
        Creates the index if not exists.
        """
        if not self.index_exists():
            if self._mappings:
                mappings = mappings_to_dict(self._mappings)
            else:
                mappings = None
            if self._settings:
                settings = self._settings
            else:
                settings = None
            resp = self._es.indices.create(index=self._index, mappings=mappings, settings=settings)
            return resp.body

    def index_update(self):
        """
        Update index mappings.
        """
        if not self.index_exists():
            self.index_create()
        else:
            if self._mappings:
                try:
                    mappings = mappings_to_dict(self._mappings)
                    resp = self._es.indices.put_mapping(index=self._index, **mappings)
                    return resp.body
                except RequestError as e:
                    logging.warning(f"Could not update mappings of index '{self._index}' ({e})")

    def index_rebuild(self):
        """
        Rebuild index (should fix mapping errors).
        """
        next_index = self._index + '-' + datetime.now().strftime("%Y%m%d%H%M%S%f")

        if self._mappings:
            mappings = mappings_to_dict(self._mappings)
        else:
            mappings = None
        if self._settings:
            settings = self._settings
        else:
            settings = None

        # create new index
        self._es.indices.create(index=next_index, mappings=mappings, settings=settings)

        # move data
        self._es.reindex(
            dest={"index": next_index},
            source={"index": self._index},
            # request_timeout=3600
        )

        # refresh the index to make the changes visible
        self._es.indices.refresh(index=next_index)

        # a) move alias (without downtime)
        aliases = self._es.indices.get_alias(index=self._index, ignore_unavailable=True)
        if aliases and self._index not in aliases:
            alias = self._index
            index = next(iter(aliases.keys()))
            self._es.indices.update_aliases(actions=[
                {"remove": {"alias": alias, "index": index}},
                {"add": {"alias": alias, "index": next_index}},
            ])
            self._es.indices.delete(index=index)
        # b) delete old index and set index alias on new index (with downtime)
        else:
            self._es.indices.delete(index=self._index)
            self._es.indices.update_aliases(actions=[
                {"add": {"alias": self._index, "index": next_index}},
            ])

    def index_copy(self, index_name: str):
        """
        Rebuild index (should fix mapping errors).
        """
        if self._mappings:
            mappings = mappings_to_dict(self._mappings)
        else:
            mappings = None
        if self._settings:
            settings = self._settings
        else:
            settings = None

        # create new index
        self._es.indices.create(index=index_name, mappings=mappings, settings=settings)

        # move data
        self._es.reindex(
            dest={"index": index_name},
            source={"index": self._index}
            # request_timeout=3600
        )

        # refresh the index to make the changes visible
        self._es.indices.refresh(index=index_name)
        return Store(self._es, index_name)

    def index_delete(self):
        """
        Deletes the index.
        """
        try:
            resp = self._es.indices.delete(index=self._index)
            return resp.body
        except NotFoundError:
            return None

    def has(self, id: str) -> bool:
        """
        Checks whether the document exists in the index.
        """
        try:
            resp = self._es.exists(index=self._index, id=id)
            return resp.body
        except NotFoundError:
            return False

    def __contains__(self, id: str) -> bool:
        return self.has(id)

    def get(self, id: str, **kwargs) -> T:
        """
        Returns doc.
        """
        try:
            res = self._es.get(index=self._index, id=id, **kwargs)
        except NotFoundError:
            raise KeyError(id) from None

        if not res["found"]:
            raise KeyError(id)

        hit = Hit(res)
        doc = cast(Dict, hit.to_dict())
        return self._from_dict(doc)

    def __getitem__(self, id: str) -> T:
        return self.get(id)

    def put(
        self, id: str, item: T, refresh: Union[bool, Literal['wait_for']] = False
    ):
        return self.update(id, item, refresh=refresh)

    def __setitem__(self, id: str, item: T):
        self.put(id, item)

    def delete(
        self, id: str, refresh: Union[bool, Literal['wait_for']] = False
    ):
        """
        Removes a document from the index.
        """
        try:
            resp = self._es.delete(index=self._index, id=id, refresh=refresh)
            return resp.body
        except NotFoundError:
            raise KeyError(id) from None

    def __delitem__(self, id: str):
        self.delete(id)

    @ensure_index_exists()
    def update(
        self, id: str, item: T, refresh: Union[bool, Literal['wait_for']] = False
    ):
        """
        Creates or updates a document in the index.
        """
        doc = self._to_dict(item)
        resp = self._es.index(index=self._index, document=doc, id=id, refresh=refresh)
        return resp.body

    @ensure_index_exists()
    def upsert(
        self,
        id: str,
        item: Union[Dict[str, Any], Any],
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
                    resp = self._es.update(index=self._index,
                                           id=id,
                                           script=script,
                                           scripted_upsert=True,
                                           upsert={},
                                           retry_on_conflict=retry,
                                           refresh=refresh)
                else:
                    resp = self._es.update(index=self._index,
                                           id=id,
                                           script=script,
                                           retry_on_conflict=retry,
                                           refresh=refresh)
            else:
                if create:
                    resp = self._es.update(index=self._index,
                                           id=id,
                                           doc=doc,
                                           doc_as_upsert=True,
                                           retry_on_conflict=retry,
                                           refresh=refresh)
                else:
                    resp = self._es.update(index=self._index,
                                           id=id,
                                           doc=doc,
                                           retry_on_conflict=retry,
                                           refresh=refresh)
            return resp.body

        except NotFoundError:
            raise KeyError(id) from None

    @ensure_index_exists()
    def bulk_delete(
        self,
        ids: Union[Iterable[str], Iterable[str]],
        stats_only: bool = False,
        refresh: Union[bool, Literal['wait_for']] = False
    ):
        """
        Remove documents from the index.
        """
        def iter_actions():
            for id in ids:
                yield dict(_index=self._index, _id=id, _op_type='delete')

        return bulk(self._es, iter_actions(), raise_on_error=False, stats_only=stats_only, refresh=refresh)

    @ensure_index_exists()
    def bulk_update(
        self,
        items: Iterable[Tuple[str, T]],
        retry: int = DEFAULT_RETRY,
        stats_only: bool = False,
        refresh: Union[bool, Literal['wait_for']] = False
    ):
        """
        Updates documents in the index.
        """
        def iter_actions():
            for key, item in items:
                doc = self._to_dict(item)
                # use _source=doc and not **doc as it allows fields with reserved names like "version"
                yield dict(_index=self._index, _id=key, _op_type='index', retry_on_conflict=retry, _source=doc)

        return bulk(self._es, iter_actions(), raise_on_error=False, stats_only=stats_only, refresh=refresh)

    @ensure_index_exists()
    def bulk_upsert(
        self,
        items: Iterable[Tuple[str, Union[Dict[str, Any], Any]]],
        source: Optional[str] = None,
        create: bool = False,
        retry: int = DEFAULT_RETRY,
        stats_only: bool = False,
        refresh: Union[bool, Literal['wait_for']] = False
    ):
        """
        Updates documents in the index with partial content.
        Interprets items as script params if a script source is given.
        """
        def iter_actions():
            for key, item in items:
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

        return bulk(self._es, iter_actions(), raise_on_error=False, stats_only=stats_only, refresh=refresh)

    @ensure_index_exists()
    def delete_by_query(self, query: Dict, refresh: Optional[bool] = False):
        """
        Deletes documents matching the provided query.
        """
        resp = self._es.delete_by_query(index=self._index,
                                        query=query,
                                        conflicts='proceed',
                                        refresh=refresh)
        return resp.body

    @ensure_index_exists()
    def update_by_query(
        self,
        query: Dict,
        item: Union[Dict[str, Any], Any],
        source: str = UPDATE_SCRIPT,
        refresh: Optional[bool] = False,
        **kwargs
    ):
        """
        Update documents matching the provided query.
        Interprets items as script params.
        """
        doc = self._to_dict(item)
        resp = self._es.update_by_query(index=self._index,
                                        script=dict(source=source, params=doc),
                                        query=query,
                                        conflicts='proceed',
                                        refresh=refresh,
                                        **kwargs)
        return resp.body

    def mget(
        self, ids: Union[Iterable[str], Iterable[str]], chunk_size: int = 500, **kwargs
    ) -> Generator[Tuple[str, Optional[T]], None, None]:
        for chunk in chunked(ids, chunk_size):
            resp = self._es.mget(index=self._index, ids=chunk, **kwargs)
            if "docs" in resp:
                for resp in resp["docs"]:
                    if resp.get("found", False):
                        doc = cast(Dict, Hit(resp).to_dict())
                        yield resp["_id"], self._from_dict(doc)
                    else:
                        yield resp["_id"], None

    def refresh(self):
        resp = self._es.indices.refresh(index=self._index)
        return resp.body

    def flush(self):
        resp = self._es.indices.flush(index=self._index)
        return resp.body

    def __iter__(self):
        return self.keys()

    def keys(self):
        try:
            for resp in scan(self._es, index=self._index, _source=False):
                assert isinstance(resp, dict)
                yield resp["_id"]
        except NotFoundError:
            pass

    def values(self):
        for _, item in self.items():
            yield item

    def items(self, query: Optional[Dict] = None, **kwargs) -> Generator[Tuple[str, T], None, None]:
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
            for resp in scan(self._es, index=self._index, query=query, **kwargs):
                assert isinstance(resp, dict)
                doc = cast(Dict, Hit(resp).to_dict())
                yield resp["_id"], self._from_dict(doc)
        except NotFoundError:
            pass

    def count(self, query: Optional[Dict] = None) -> int:
        try:
            resp = self._es.count(index=self._index, query=query)
            return resp.body['count']
        except NotFoundError:
            return 0

    def __len__(self):
        return self.count()

    def search(self, query: Dict, **kwargs):
        try:
            resp = self._es.search(index=self._index, **query, **kwargs)
            return resp.body
        except NotFoundError:
            return None

    def filter_by(
        self,
        fields: Dict[str, Union[Union[str, int, float, bool], List[Union[str, int, float, bool]]]]
    ) -> List[T]:
        return [item for _, item in self.items(query={
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
