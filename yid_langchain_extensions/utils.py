from typing import Annotated, Type, Any

from pydantic import ValidationError, PlainValidator
from pydantic.v1 import BaseModel as BaseModelV1


def validated(value: BaseModelV1, target_class: Type[Any]) -> BaseModelV1:
    if isinstance(value, target_class):
        return value
    raise ValidationError("Input of invalid type")


def pydantic_v1_port(cls):
    return Annotated[cls, PlainValidator(lambda val: validated(val, cls))]
