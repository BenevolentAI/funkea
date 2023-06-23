import pytest

from funkea.core.utils import files


@pytest.fixture(scope="function", autouse=True)
def mock_parse_file(monkeypatch):
    monkeypatch.setattr(
        "funkea.core.utils.files.FileRegistry.parse_file",
        lambda *args, **kwargs: files.FileRegistry(),
    )


@pytest.fixture(scope="function")
def mock_is_zipfile(monkeypatch):
    monkeypatch.setattr("funkea.core.utils.files.zipfile.is_zipfile", lambda *args, **kwargs: True)


@pytest.fixture(scope="function")
def mock_zipfile_read(monkeypatch):
    class _MockZipfile:
        def __init__(self, *args, **kwargs):
            pass

        def open(self, *args, **kwargs):
            return self

        def read(self):
            return '{"gtex": "some/path"}'

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    monkeypatch.setattr("funkea.core.utils.files.zipfile.ZipFile", _MockZipfile)


def test_home():
    assert files.HOME.exists()
    assert files.HOME.stem == "funkea"


def test_resources():
    assert files.RESOURCES.exists()
    assert files.RESOURCES.is_dir()


class TestFileRegistry:
    def test_default(self):
        fr = files.FileRegistry.default()
        assert fr == files.FileRegistry()

    def test_zipped_default(self, mock_is_zipfile, mock_zipfile_read):
        fr = files.FileRegistry.default()
        assert fr.gtex == "some/path"
