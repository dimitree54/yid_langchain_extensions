from typing import Union, Dict, Any, Type, Callable, List

from langchain.agents.agent import MultiActionAgentOutputParser
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.agents import AgentFinish, AgentAction
from langchain_core.outputs import Generation
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool, _rm_titles, \
    convert_to_openai_function
from langchain_core.utils.json_schema import dereference_refs
from pydantic import BaseModel as BaseModelV2
from pydantic.v1 import BaseModel as BaseModelV1
from pydantic_core import ValidationError


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


class PydanticOutputParser(MultiActionAgentOutputParser):
    pydantic_class: Union[Type[BaseModelV1], Type[BaseModelV2]]
    base_parser: MultiActionAgentOutputParser = OpenAIToolsAgentOutputParser()
    return_key: str = "pydantic_objects"

    @property
    def _type(self) -> str:
        return f"pydantic-{self.base_parser._type}"

    def parse_result(
            self, result: List[Generation], *, partial: bool = False
    ) -> AgentFinish:
        tool_call_actions = self.base_parser.parse_result(result, partial=partial)
        if isinstance(tool_call_actions, AgentFinish):
            return tool_call_actions
        try:
            return AgentFinish(
                return_values={
                    self.return_key: [self.pydantic_class(**agent_action.tool_input) for agent_action in
                                      tool_call_actions]
                },
                log="\n".join([agent_action.log for agent_action in tool_call_actions])
            )
        except ValidationError as e:
            raise ValueError(f"LLM failed to predict proper structure for pydantic class:\n{str(e)}")

    def parse(self, text: str) -> AgentFinish:
        raise ValueError("Can only parse messages")


class ThoughtStrippingParser(MultiActionAgentOutputParser):
    thought_name: str
    base_parser: MultiActionAgentOutputParser

    @property
    def _type(self) -> str:
        return f"thought-stripping{self.base_parser._type}"

    def parse_result(
            self, result: List[Generation], *, partial: bool = False
    ) -> Union[List[AgentAction], AgentFinish]:
        tool_call_actions = self.base_parser.parse_result(result, partial=partial)
        if isinstance(tool_call_actions, AgentFinish):
            return tool_call_actions
        return [agent_action for agent_action in tool_call_actions if agent_action.name != self.thought_name]

    def parse(self, text: str) -> AgentFinish:
        raise ValueError("Can only parse messages")
