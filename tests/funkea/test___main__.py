import pytest

from funkea import __main__
from funkea.core.utils import files


@pytest.fixture(scope="function", autouse=True)
def mock_input(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *args, **kwargs: "")


@pytest.fixture(scope="function", autouse=True)
def mock_open(monkeypatch):
    class _MockIO:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def write(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: _MockIO())


@pytest.fixture(scope="function")
def mock_is_zipfile(monkeypatch):
    monkeypatch.setattr("funkea.core.utils.files.zipfile.is_zipfile", lambda *args, **kwargs: False)


def test__interactive():
    res = __main__._interactive()
    files.FileRegistry.parse_obj(res)
    assert res == files.FileRegistry().dict()


def test__from_stem():
    res = __main__._from_stem("some_stem")
    files.FileRegistry.parse_obj(res)
    assert all(f"some_stem/{k}" == v for k, v in res.items())


def test__store():
    __main__._store(files.FileRegistry())


class TestCLI:
    def test_init(self):
        __main__.CLI().init()
        __main__.CLI().init(from_stem="some_stem")

    def test_set(self, mock_is_zipfile):
        __main__.CLI().set("gtex", "my_path")

    def test_reset(self):
        with pytest.raises(ValueError, match="invalid truth value"):
            __main__.CLI().reset()
        __main__.CLI().reset(True)
