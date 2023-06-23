import pytest

from funkea.components import locus_definition, variant_selection
from funkea.core import pipeline


@pytest.fixture(scope="function", autouse=True)
def mock_abc(mock_abstract_methods):
    mock_abstract_methods(pipeline.DataPipeline)


class TestDataPipeline:
    def test_init(self):
        pipeline.DataPipeline(locus_definition.Identity())

    def test_get_variant_selector(self):
        pipe = pipeline.DataPipeline(locus_definition.Identity())
        assert isinstance(pipe.get_variant_selector(), variant_selection.Identity)
        pipe.variant_selector = variant_selection.AssociationThreshold()
        assert isinstance(pipe.get_variant_selector(), variant_selection.AssociationThreshold)
