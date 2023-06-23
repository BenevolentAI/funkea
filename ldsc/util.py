"""LDSC utility functions and decorators."""
import enum
import inspect
import os
import warnings
from collections import namedtuple
from functools import WRAPPER_ASSIGNMENTS, partial, update_wrapper, wraps
from importlib import import_module
from types import ModuleType
from typing import Any, Callable, ParamSpec, TypeVar

import jax
import pandas as pd

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


class Backend(enum.Enum):
    JAX: str = "jax.numpy"
    NUMPY: str = "numpy"


def _get_backend_impl(identifier: str) -> Backend:
    """Get the linear algebra backend enum identifier.

    Args:
        identifier: a string representation of the backend. It is *not* case-sensitive and
            also not sensitive to flanking whitespace.

    Returns:
        The backend enum instance.
    """
    ident = identifier.strip().upper()
    try:
        # here we want to maximise robustness by removing accidental whitespace and enforcing
        # all-caps moreover, the explicit enum gives us the ability to adequately surface an error
        # related to an unsupported option
        bck = Backend[ident]
    except KeyError:
        msg = ", ".join(map(getattr, Backend, ("name",) * len(Backend)))
        raise ValueError(f"{ident!r} is not a valid backend. Should be one of: {msg}")

    return bck


def backend(env_var: str = "LDSC_BACKEND") -> ModuleType:
    """Get the linear algebra backend engine for ``ldsc``. Accesses the state via an environment
    variable defined by ``env_var``. One of: NUMPY, JAX.

    Args:
        env_var: The environment variable to query to get the backend to use.
            (``default="LDSC_BACKEND"``)

    Returns:
        linear algebra backend.

    Warnings:
        JAX backend has been minimally tested.
    """
    # seems a bit nuts but perhaps smart?
    bck = _get_backend_impl(os.environ.get(env_var, "NUMPY"))
    if bck == Backend.JAX:
        warnings.warn(
            "Jax backend has been minimally tested.",
            UserWarning,
        )
    module = import_module(bck.value)
    if bck == Backend.JAX and os.environ.get("JAX_DBL_PRC"):
        # jax defaults to single precision floats, while numpy defaults to double precision. In case
        # someone wants double precision, one only needs to set the JAX_DBL_PRC environment variable
        from jax import config

        config.update("jax_enable_x64", True)

    return module


np = backend()


def _set_fun_signature(fn: Callable[P, R]) -> Callable[P, R]:
    """Set the function signature attribute. Necessary to not break Spark when decorating UDFs.

    Args:
        fn: The function for which the signature need be set.

    Returns:
        ``fn`` but with the ``__signature__`` attribute set.
    """
    signature = inspect.signature(fn)
    if not hasattr(fn, "__signature__"):
        fn.__signature__ = None  # type: ignore
    fn.__signature__ = fn.__signature__ or signature  # type: ignore
    return fn


def populate_arguments_from_locals(fn: Callable[P, R]) -> Callable[[dict[str, Any]], R]:
    params = inspect.signature(fn).parameters

    def _fn(var_dict: dict[str, Any]):
        call = {name: value for name, value in var_dict.items() if name in params}
        out = fn(**call)
        return out

    return _fn


def instance_has_attr(
    attribute: str,
    exception: Exception,
    *,
    raise_if_none: bool = False,
    raise_if_not_truthy: bool = False,
) -> Callable:
    """Check if the object instance has a particular attribute. If not, raise an exception. This is
    useful for methods, which can only be called if a certain attribute has been set (e.g. accessing
    coefficients from a fit model).

    Args:
        attribute: The name of the object attribute.
        exception: The exception to be raised.
        raise_if_none: Whether to raise if the instance attribute is None. (``default=False``)
        raise_if_not_truthy: Whether to raise if the instance attribute is not "truthy".
            (``default=False``)

    Returns:
        The wrapped method.
    """

    def decorator(method):
        method = _set_fun_signature(method)

        def _format_exc_args(*args):
            args = tuple(
                arg.format(name=method.__qualname__) if isinstance(arg, str) else arg
                for arg in args
            )
            return args

        @wraps(method, assigned=WRAPPER_ASSIGNMENTS)
        def _method(self, *args, **kwargs):
            if (
                not hasattr(self, attribute)
                or (raise_if_none and getattr(self, attribute) is None)
                or (raise_if_not_truthy and not getattr(self, attribute))
            ):
                exception.args = _format_exc_args(*exception.args)
                raise exception
            out = method(self, *args, **kwargs)
            return out

        return _method

    return decorator


def check_fit_fn_arrays(method=None, *, allow_multiple_outputs: bool = False):
    """Check the fit function input arrays. Checks shape, type etc..

    Args:
        method: the fit function to be decorated. (``default=None``)
        allow_multiple_outputs: Whether to allow 2-d ``y`` array.

    Returns:
        The wrapped fit function.
    """
    if method is None:
        return update_wrapper(
            partial(check_fit_fn_arrays, allow_multiple_outputs=allow_multiple_outputs),
            check_fit_fn_arrays,
        )

    method = _set_fun_signature(method)

    @wraps(method, assigned=WRAPPER_ASSIGNMENTS)
    def _method(self, X, y, *args, **kwargs):
        if len(X.shape) != 2:
            raise ValueError(f"`X` must be a 2-d array, got {len(X.shape)}-d")
        if not allow_multiple_outputs and len(y.shape) > 1:
            raise ValueError(f"`y` must be a 1-d array, got {len(y.shape)}-d")
        if len(X) != len(y):
            raise ValueError(
                f"`X` and `y` must be of same length, got {len(X)} and {len(y)}, respectively"
            )
        out = method(self, X, y, *args, **kwargs)
        return out

    return _method


def _enforce_tuple(val) -> tuple:
    """Enforce that any given value is a tuple."""
    if not isinstance(val, tuple):
        return (val,)
    return val


def enforce_data_dim(n_dim: int | tuple[int, ...]):
    """Enforce that the ``X`` in the method has a certain dimensionality, i.e. that the second
    dimension of the matrix is of a given (set of) length(s). If ``X`` has a ``X.shape[1]`` not
    allowed by ``n_dim`` a ``NotImplementedError`` exception will be raised.

    Args:
        n_dim: Either an int or a tuple of ints. Specifies the allowed data dimensionalities.

    Returns:
        The function decorator.

    Raises:
        NotImplementedError
    """
    _n_dim = _enforce_tuple(n_dim)

    def decorator(method):
        method = _set_fun_signature(method)
        method.__doc__ = (
            (method.__doc__ or "")
            + f"""
        Note:
            `X` feature dimension must be one of: {", ".join(map(str, _n_dim))}
        """
        )

        @wraps(method, assigned=WRAPPER_ASSIGNMENTS)
        def _method(self, X, *args, **kwargs):
            (_, d) = X.shape
            if d not in _n_dim:
                msg = ", ".join(map(str, _n_dim))
                raise NotImplementedError(
                    f"{self.__class__.__name__} has no implementation for data of dimension {d}. "
                    f"Only supports {msg}."
                )
            out = method(self, X, *args, **kwargs)
            return out

        return _method

    return decorator


def enforce_array_types(fn: Callable[P, R]) -> Callable[P, R]:
    """Decorator enforcing that all inputs are of array type.

    Args:
        fn: A function where we want to ensure array inputs.

    Returns:
        The same function as ``fn``, only that all inputs are enforced as arrays.
    """
    from typing import Iterable

    fn = _set_fun_signature(fn)

    def _to_arr(val):
        if isinstance(val, np.ndarray):
            return val
        if isinstance(val, Iterable) and not isinstance(val, str):
            return np.array(val)
        return np.array([val])

    @wraps(fn, assigned=WRAPPER_ASSIGNMENTS)
    def _fn(*args: P.args, **kwargs: P.kwargs) -> R:
        out = fn(*tuple(_to_arr(arg) for arg in args), **{k: _to_arr(v) for k, v in kwargs.items()})
        return out

    return _fn


def harmonise_inputs_to_backend(method: Callable):
    """Force all arrays to correspond to the desired linear algebra ``backend``.

    Args:
        method: The class method to be wrapped. Only allows methods for now.

    Returns:
        The decorated function, which ensures that all input arrays will be from the
            desired backend.
    """
    method = _set_fun_signature(method)

    @wraps(method, assigned=WRAPPER_ASSIGNMENTS)
    def _method(self, X, y, *args, **kwargs):
        X = np.array(X)
        y = np.array(y)
        out = method(self, X, y, *args, **kwargs)
        return out

    return _method


def remove_kw(kw: dict, *, remove: tuple = ()) -> dict:
    """Convenience function to exclude a set of key-value pairs from a given (flat) dictionary.

    Args:
        kw: A flat dictionary.
        remove: The set of keys to remove from the dict.

    Returns:
        The same dictionary minus the key-value pairs specified by ``remove``.
    """
    out = {k: v for k, v in kw.items() if k not in remove}
    return out


def remove_self_kw(kw: dict) -> dict:
    """Convenience function for removing the ``self`` key-value pair from a (flat) dictionary.

    Useful inside method scopes.

    Args:
        kw: Flat dictionary.

    Returns:
        The dictionary without the ``self`` key-value pair.
    """
    # "self" is a nuisance variable when we want to pass local scopes between instance methods,
    # hence we filter it out
    out = remove_kw(kw, remove=("self",))
    return out


def flatten_output_array(fn: Callable[P, np.ndarray]) -> Callable[P, np.ndarray]:
    """A decorator flattening an output array.

    Args:
        fn: A callable returning an array.

    Returns:
        Function with output array flattening.
    """

    @wraps(fn, assigned=WRAPPER_ASSIGNMENTS)
    def _fn(*args: P.args, **kwargs: P.kwargs) -> np.ndarray:
        out = fn(*args, **kwargs)
        return out.reshape(-1)

    return _fn


def enforce_array_shape(
    argnum: int, shape: tuple, condition: Callable[[np.ndarray], bool]
) -> Callable:
    """Decorator factory used to enforce a certain input array shape, upon a condition.

    Args:
        argnum: The index position of the argument to be checked.
        shape: The desired array shape.
        condition: Callable taking an array and returning a boolean.
            The array is the specified argument and if ``condition`` evaluates to true, the array is
            reshaped into the desired shape.

    Returns:
        The function decorator.
    """

    def decorator(fn: Callable):
        fn = _set_fun_signature(fn)
        arg_info = _get_argument_info(fn, argnum)

        @wraps(fn, assigned=WRAPPER_ASSIGNMENTS)
        def _fn(*args, **kwargs):
            array, arg_type = _get_argument(arg_info, args, kwargs)
            if condition(array):
                array = array.reshape(shape)
            args, kwargs = _put_argument(arg_info, arg_type, array, args, kwargs)
            out = fn(*args, **kwargs)
            return out

        return _fn

    return decorator


def enforce_column_vector_if_1d(argnum: int) -> Callable:
    """Enforces a column vector if the array is 1-d."""
    return enforce_array_shape(argnum, shape=(-1, 1), condition=lambda arr: len(arr.shape) == 1)


def add_docstring(fn=None, doc: str = "") -> Callable:
    """Add / extend a docstring to a function.

    Args:
        fn: The function whose docstring will be extended.
        doc: The docstring to be added.

    Returns:
        The original function with additional docs.
    """
    if not callable(fn):
        return update_wrapper(partial(add_docstring, doc=doc), add_docstring)

    fn.__doc__ = (fn.__doc__ or "") + "\n" + doc
    return fn


class Argument(enum.Enum):
    POSITIONAL = 0
    KEYWORD = 1
    DEFAULT = 2


ArgumentInfo = namedtuple(
    "ArgumentInfo", ["argnum", "argname", "default"], defaults=(inspect.Parameter.empty,)
)


def _get_argument_info(fn: Callable, argnum: int) -> ArgumentInfo:
    """Get the concise information needed to extract a given argument from a set of args and kwargs.

    Args:
        fn: Function for which one wants to get a specific argument's information.
        argnum: The positional index of the argument. Can be negative.

    Returns:
        A named tuple, containing the (updated) argnum, the argname and the
            parameter default (if one exists).
    """
    params = inspect.signature(fn).parameters
    if argnum < 0:
        argnum = len(params) + argnum

    argname = list(params)[argnum]
    return ArgumentInfo(argnum, argname, params[argname].default)


def _get_argument(argument_info: ArgumentInfo, args: tuple, kwargs: dict) -> tuple[Any, Argument]:
    """Get the value and the way by which the argument was passed from a set of args and kwargs.

    Args:
        argument_info: The namedtuple containing the argnum, argname and argument default.
        args: the args tuple passed into a function.
        kwargs: the kwargs dict passed into a function.

    Returns:
        A tuple containing the argument value and an enum defining how it was passed into the
        function.
    """
    argnum, argname, default = argument_info
    if argname in kwargs:
        return kwargs[argname], Argument.KEYWORD
    if len(args) > argnum:
        return args[argnum], Argument.POSITIONAL
    return default, Argument.DEFAULT


def _put_argument(
    argument_info: ArgumentInfo, arg_type: Argument, argument: Any, args: tuple, kwargs: dict
) -> tuple[tuple, dict]:
    """Put an argument into a set of args and kwargs.

    Args:
        argument_info: a namedtuple containing the argnum, argname and argument default.
        arg_type: The enum defining where the argument should be placed (args or kwargs).
        argument: The argument to be placed in args and kwargs.
        args: The args tuple.
        kwargs: The kwargs dict.

    Returns:
        The args tuple and kwargs dict.
    """
    argnum, argname, default = argument_info
    if arg_type == Argument.DEFAULT:
        kwargs[argname] = default
    elif arg_type == Argument.KEYWORD:
        kwargs[argname] = argument
    elif arg_type == Argument.POSITIONAL:
        args = tuple(arg if i != argnum else argument for i, arg in enumerate(args))
    return args, kwargs


class ToPandasMixin:
    """A mixin class for casting a dataclass to a pandas dataframe, where each class attribute is
    transformed into a column."""

    def to_pandas(self, array_to_item: bool = True) -> pd.DataFrame:
        """Cast the dataclass object into a pandas dataframe.

        Args:
            array_to_item: Whether to get the array ``.item()``. (``default=True``)

        Returns:
            The dataclass as a pandas dataframe.
        """
        items = self.__dict__
        if array_to_item:

            def _to_item(e):
                if isinstance(e, np.ndarray):
                    return e.item()
                return e

            items = jax.tree_map(_to_item, items)
        return pd.DataFrame.from_dict(items, orient="index").transpose()
