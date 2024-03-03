from typing import Annotated, Type, Any

from pydantic import ValidationError, PlainValidator
from pydantic.v1 import BaseModel as BaseModelV1, parse_obj_as


def validated(value: BaseModelV1, target_class: Type[Any]) -> BaseModelV1:
    try:
        return parse_obj_as(target_class, value)  # noqa
    except ValidationError as e:
        raise ValidationError(f"Input of invalid type: {e}")


def pydantic_v1_port(cls):
    return Annotated[cls, PlainValidator(lambda val: validated(val, cls))]
