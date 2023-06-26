import platform
import warnings
from importlib import import_module
from typing import Union

import numpy as np
import pandas as pd
import pytest

import ldsc.util


class Cases:
    ENV = [
        ({"LDSC_BACKEND": "NUMPY"}, "numpy"),
        ({"LDSC_BACKEND": "JAX"}, "jax.numpy"),
        ({"LDSC_BACKEND": "JAX", "JAX_DBL_PRC": "true"}, "jax.numpy"),
        ({"LDSC_BACKEND": "NUMPY", "JAX_DBL_PRC": "true"}, "numpy"),
    ]
    FUNCTIONS = [
        (lambda: (None,), ("empty",)),
        (lambda x: (x,), ("x",)),
        (lambda x, y=1: (x, y), ("x", "y")),
    ]
    VALS = [1, (1, 2), 2, [1, 2, 3]]


@pytest.fixture()
def mock_env(monkeypatch):
    def _path(env):
        monkeypatch.setattr("ldsc.util.os.environ", env)

    return _path


class TestBackend:
    def test__get_backend_impl(self):
        assert ldsc.util._get_backend_impl("numpy") == ldsc.util.Backend.NUMPY
        assert ldsc.util._get_backend_impl("jax") == ldsc.util.Backend.JAX

        with pytest.raises(ValueError):
            ldsc.util._get_backend_impl("openblas")

    @pytest.mark.parametrize(["env", "module"], Cases.ENV)
    def test_backend(self, mock_env, env, module):
        uname = platform.uname()
        if uname.system == "Linux" and uname.machine == "aarch64" and module == "jax.numpy":
            pytest.skip("JAX is not supported on ARM")
        mock_env(env)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            assert ldsc.util.backend() is import_module(module)

        try:
            from jax import config
        except (ImportError, ModuleNotFoundError):
            config = None

        if env["LDSC_BACKEND"] == "JAX":
            if config is None:
                pytest.skip("JAX is not installed")
            with pytest.warns(
                UserWarning,
                match="Jax backend has been minimally tested.",
            ):
                ldsc.util.backend()

        assert (
            "JAX_DBL_PRC" not in env
            or env["LDSC_BACKEND"] == "NUMPY"
            or config.read("jax_enable_x64")
        )


@pytest.mark.parametrize("fun", [f for f, _ in Cases.FUNCTIONS])
def test__set_fun_signature(fun):
    assert not hasattr(fun, "__signature__")
    fun = ldsc.util._set_fun_signature(fun)
    assert hasattr(fun, "__signature__")


@pytest.mark.parametrize(["fun", "names"], Cases.FUNCTIONS)
def test_populate_arguments_from_locals(fun, names):
    fun = ldsc.util.populate_arguments_from_locals(fun)

    empty = None
    x = 1
    y = 1

    lcs = locals()
    assert fun(lcs) == tuple(lcs[k] for k in names)


def test_instance_has_attr():
    class _A:
        x: int

        @ldsc.util.instance_has_attr("x", ValueError())
        def f(self):
            return self.x

    a = _A()
    with pytest.raises(ValueError):
        a.f()

    a.x = 1
    assert a.f() == 1

    class _B:
        x: int

        @ldsc.util.instance_has_attr("x", ValueError(), raise_if_none=True)
        def g(self):
            return self.x

    b = _B()
    with pytest.raises(ValueError):
        b.g()

    b.x = 1
    assert b.g() == 1

    class _C:
        x = ""

        @ldsc.util.instance_has_attr("x", ValueError(), raise_if_not_truthy=True)
        def h(self):
            return self.x

    c = _C()
    with pytest.raises(ValueError):
        c.h()

    c.x = "hello"
    assert c.h() == "hello"


def test_check_fit_fn_arrays():
    class _K:
        @ldsc.util.check_fit_fn_arrays
        def fit(self, X, y):
            pass

    k = _K()
    X = np.random.randn(10, 1)
    y = np.random.randn(10)

    k.fit(X, y)
    with pytest.raises(ValueError, match=r"`X` must be a 2-d array, got \d-d"):
        k.fit(X.flatten(), y)

    with pytest.raises(ValueError, match=r"`y` must be a 1-d array, got \d-d"):
        k.fit(X, y.reshape((-1, 1)))

    with pytest.raises(
        ValueError, match=r"`X` and `y` must be of same length, got \d+ and \d+, respectively"
    ):
        k.fit(X[:5], y)


@pytest.mark.parametrize("val", Cases.VALS)
def test__enforce_tuple(val):
    assert isinstance(ldsc.util._enforce_tuple(val), tuple)


def test_enforce_data_dim():
    class _A:
        @ldsc.util.enforce_data_dim(n_dim=(1, 2))
        def f(self, X):
            return X

    _A().f(np.random.randn(10, 1))
    _A().f(np.random.randn(10, 2))

    with pytest.raises(NotImplementedError):
        _A().f(np.random.randn(10, 3))


def test_enforce_array_types():
    @ldsc.util.enforce_array_types
    def f(x, y):
        return x, y

    assert all(isinstance(a, np.ndarray) for a in f(1, 2))


def test_harmonise_inputs_to_backend():
    class _K:
        @ldsc.util.harmonise_inputs_to_backend
        def f(self, X, y):
            return X, y

    try:
        import jax.numpy as jnp
    except (ImportError, ModuleNotFoundError):
        return

    assert all(isinstance(a, np.ndarray) for a in _K().f(jnp.array([1]), jnp.array([1])))


def test_remove_kw():
    d = {"a": 1, "b": 2}
    c = ldsc.util.remove_kw(d, remove=("a",))
    assert c == {"b": 2}
    assert d is not c


def test_remove_self_kw():
    assert ldsc.util.remove_self_kw({"a": 1, "self": 2}) == {"a": 1}


def test_flatten_output_array():
    @ldsc.util.flatten_output_array
    def f(x):
        return x

    assert len(f(np.random.randn(10)).shape) == 1
    assert len(f(np.random.randn(10, 1)).shape) == 1


def test_enforce_array_shape():
    @ldsc.util.enforce_array_shape(argnum=0, shape=(-1, 1), condition=lambda _: True)
    def f(x):
        return x

    assert len(f(np.random.randn(10)).shape) == 2
    assert f(np.random.randn(10)).shape[-1] == 1


def test_enforce_column_vector_if_1d():
    @ldsc.util.enforce_column_vector_if_1d(argnum=0)
    def f(x):
        return x

    assert len(f(np.random.randn(10)).shape) == 2
    assert len(f(np.random.randn(10, 1)).shape) == 2
    assert len(f(np.random.randn(10, 1, 1)).shape) == 3


def test_add_docstring():
    def f():
        """Hello."""

    f = ldsc.util.add_docstring(f, "world")

    assert f.__doc__ == "Hello.\nworld"


class TestToPandasMixin:
    def test_to_pandas(self):
        from dataclasses import dataclass

        @dataclass
        class A(ldsc.util.ToPandasMixin):
            a: Union[int, np.ndarray]
            b: Union[int, np.ndarray]

        assert isinstance(A(1, 2).to_pandas(), pd.DataFrame)
        assert isinstance(A(np.array([1]), 1).to_pandas(array_to_item=True), pd.DataFrame)
