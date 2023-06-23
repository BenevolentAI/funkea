import pytest

from funkea.core import method


@pytest.fixture(scope="function", autouse=True)
def mock_abc(mock_abstract_methods):
    mock_abstract_methods(method.EnrichmentMethod)


class TestEnrichmentMethod:
    def test_init(self, mock_abstract_methods):
        # there is no real functionality here, so not much to test
        method.EnrichmentMethod()
