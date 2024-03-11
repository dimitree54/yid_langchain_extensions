from copy import deepcopy
from typing import Annotated, Type, Any, Dict, Union, Callable, Optional

from langchain_core.runnables import RunnableBinding, RunnableConfig
from langchain_core.runnables.utils import Input, Output
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import _rm_titles, convert_to_openai_function, convert_to_openai_tool
from langchain_core.utils.json_schema import dereference_refs
from pydantic import ValidationError, PlainValidator, BaseModel as BaseModelV2
from pydantic.v1 import BaseModel as BaseModelV1, parse_obj_as


def validated(value: BaseModelV1, target_class: Type[Any]) -> BaseModelV1:
    try:
        return parse_obj_as(target_class, value)  # noqa
    except ValidationError as e:
        raise ValidationError(f"Input of invalid type: {e}")


def pydantic_v1_port(cls):
    return Annotated[cls, PlainValidator(lambda val: validated(val, cls))]


def convert_pydantic_to_openai_function_v2(
        model: Type[BaseModelV2],
) -> Dict[str, Any]:
    schema = dereference_refs(model.model_json_schema())
    schema.pop("definitions", None)
    title = schema.pop("title", "")
    default_description = schema.pop("description", "")
    return {
        "name": title,
        "description": default_description,
        "parameters": _rm_titles(schema)
    }


def convert_to_openai_function_v2(
        function: Union[Dict[str, Any], Type[BaseModelV1], Callable, BaseTool, Type[BaseModelV2]],
) -> Dict[str, Any]:
    if isinstance(function, type) and issubclass(function, BaseModelV2):
        return convert_pydantic_to_openai_function_v2(function)
    else:
        return convert_to_openai_function(function)


def convert_to_openai_tool_v2(
        tool: Union[Dict[str, Any], Type[BaseModelV1], Callable, BaseTool, Type[BaseModelV2]],
) -> Dict[str, Any]:
    if isinstance(tool, type) and issubclass(tool, BaseModelV2):
        return convert_to_openai_tool(
            convert_to_openai_function_v2(tool)
        )
    else:
        return convert_to_openai_tool(tool)


class ListExtendingRunnableBinding(RunnableBinding):
    @staticmethod
    def merge_dicts_with_list_extension(base: dict, update: dict) -> dict:
        result = deepcopy(base)
        for key, value in update.items():
            if key in result and isinstance(result[key], list) and isinstance(value, list):
                result[key].extend(value)
            else:
                result[key] = value
        return result

    def invoke(
            self,
            input: Input,
            config: Optional[RunnableConfig] = None,
            **kwargs: Optional[Any],
    ) -> Output:
        return self.bound.invoke(
            input,
            self._merge_configs(config),
            **self.merge_dicts_with_list_extension(self.kwargs, kwargs),
        )
