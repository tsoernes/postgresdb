from itertools import islice
from typing import Callable, Iterable, TypeVar

from postgresdb.utils.functoolz import curried

K = TypeVar("K")
L = TypeVar("L")
V = TypeVar("V")
W = TypeVar("W")


@curried
def lmap(fn: Callable[[V], W], iterable: Iterable[V]) -> list[W]:
    """Map which returns a list"""
    return [fn(x) for x in iterable]


@curried
def chunks(iterable: Iterable[K], size: int) -> Iterable[list[K]]:
    """Yield successive chunks of a specified size from the iterable."""
    it = iter(iterable)
    item = list(islice(it, size))
    while item:
        yield item
        item = list(islice(it, size))


@curried
def filter_di_vals(pred: Callable[[V], bool], di: dict[K, V]) -> dict[K, V]:
    """Filter a dictionary by a predicate on its values."""
    return {k: v for k, v in di.items() if pred(v)}


@curried
def valuemap_dict(fn: Callable[[V], W], di: dict[K, V]) -> dict[K, W]:
    """Map a function over each value in the dict"""
    return {k: fn(v) for k, v in di.items()}


@curried
def sub_dict_inv(di, *keys):
    """Return the subset of the dictionary not matching the given keys"""
    return {k: di[k] for k in di.keys() if k not in keys}
