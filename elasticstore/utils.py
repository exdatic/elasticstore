from typing import AsyncIterable, AsyncIterator, Iterable, List, TypeVar, Union

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


async def islice(iterable: Union[Iterable[T], AsyncIterable[T]], n: int) -> AsyncIterator[T]:
    it = aiter(iterable)
    if n > 0:
        async for i in it:
            yield i
            n -= 1
            if n <= 0:
                break


async def take(n: int, iterable: Union[Iterable[T], AsyncIterable[T]]) -> List[T]:
    return [i async for i in islice(iterable, n)]


async def chunked(iterable: Union[Iterable[T], AsyncIterable[T]], n: int) -> AsyncIterable[List[T]]:
    it = aiter(iterable)
    chunk = await take(n, it)
    while chunk != []:
        yield chunk
        chunk = await take(n, it)
