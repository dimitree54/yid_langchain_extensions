from typing import Sequence, Union, Dict, Any, Type, Callable

from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool, _rm_titles, \
    convert_to_openai_function
from langchain_core.utils.json_schema import dereference_refs
from langchain_openai import ChatOpenAI
from pydantic import BaseModel as BaseModelV2
from pydantic.v1 import BaseModel as BaseModelV1


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


def create_fn_calling_llm(
        llm: ChatOpenAI,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModelV1], Type[BaseModelV2], Callable, BaseTool]]
) -> Runnable:
    llm_with_tools = llm.bind(tools=[convert_to_openai_tool_v2(tool) for tool in tools])

    agent = (
            llm_with_tools
            | OpenAIToolsAgentOutputParser()
    )
    return agent
