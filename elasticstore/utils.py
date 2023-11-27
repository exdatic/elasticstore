from typing import AsyncIterable, AsyncIterator, Iterable, Iterator, List, TypeVar, Union

T = TypeVar('T')


def aiter(iterable: Union[Iterable[T], AsyncIterable[T]]) -> AsyncIterator[T]:
    if isinstance(iterable, AsyncIterator):
        return iterable
    if isinstance(iterable, AsyncIterable):
        return iterable.__aiter__()

    async def gen() -> AsyncIterator[T]:
        for i in iterable:
            yield i
    return gen()


async def aislice(iterable: Union[Iterable[T], AsyncIterable[T]], n: int) -> AsyncIterator[T]:
    if n > 0:
        async for i in aiter(iterable):
            yield i
            n -= 1
            if n <= 0:
                break


async def atake(n: int, iterable: Union[Iterable[T], AsyncIterable[T]]) -> List[T]:
    return [i async for i in aislice(iterable, n)]


async def achunked(iterable: Union[Iterable[T], AsyncIterable[T]], n: int) -> AsyncIterable[List[T]]:
    it = aiter(iterable)
    chunk = await atake(n, it)
    while chunk != []:
        yield chunk
        chunk = await atake(n, it)


def islice(iterable: Union[Iterable[T], Iterable[T]], n: int) -> Iterator[T]:
    if n > 0:
        for i in iter(iterable):
            yield i
            n -= 1
            if n <= 0:
                break


def take(n: int, iterable: Union[Iterable[T], Iterable[T]]) -> List[T]:
    return [i for i in islice(iterable, n)]


def chunked(iterable: Union[Iterable[T], Iterable[T]], n: int) -> Iterable[List[T]]:
    it = iter(iterable)
    chunk = take(n, it)
    while chunk != []:
        yield chunk
        chunk = take(n, it)
