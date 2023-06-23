from typing import Any

import pydantic
from pyspark.ml.param import Param

PYDANTIC_VALIDATION_CONFIG: dict[str, Any] = dict(arbitrary_types_allowed=True)


class UnitFloat(pydantic.ConstrainedFloat):
    ge = 0.0
    le = 1.0


class HalfUnitFloat(UnitFloat):
    le = 0.5


ParamMap = dict[Param, Any]
