import pandas as pd
import pydantic

from funkea.core import params


class TestParams:
    def test_arbitrary_types(self):
        model = pydantic.create_model("SomeModel", foo=(pd.DataFrame, ...), __base__=params.Params)
        model(foo=pd.DataFrame())
