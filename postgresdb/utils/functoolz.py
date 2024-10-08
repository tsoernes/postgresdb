import inspect
from functools import partial, wraps
from typing import Callable


def curried(func: Callable) -> Callable:
    """Allows currying"""
    sig = inspect.signature(func)

    @wraps(func)
    def inner(*args, **kwargs):
        bind = sig.bind_partial(*args, **kwargs)
        for param in sig.parameters.values():
            if param.name not in bind.arguments and param.default is param.empty:
                # Some required arguments are missing; return a partial
                return partial(func, *args, **kwargs)
        return func(*args, **kwargs)

    return inner
