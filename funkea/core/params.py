import abc

import pydantic


class Params(pydantic.BaseModel, metaclass=abc.ABCMeta):
    """Base class for workflow parameters for default configs."""

    class Config:
        arbitrary_types_allowed = True
