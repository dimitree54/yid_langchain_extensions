from typing import Union, Type

from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.agents import AgentFinish
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import BaseOutputParser
from pydantic import BaseModel as BaseModelV2
from pydantic.v1 import BaseModel as BaseModelV1


class PydanticOutputParser(BaseOutputParser[Union[BaseModelV1, BaseModelV2]]):
    pydantic_class: Union[Type[BaseModelV1], Type[BaseModelV2]]
    base_parser: OpenAIToolsAgentOutputParser

    def parse_result(
            self, result: AIMessage, *, partial: bool = False
    ) -> Union[BaseModelV1, BaseModelV2]:
        tool_call_actions = self.base_parser.parse_result(result, partial=partial)  # noqa
        if isinstance(tool_call_actions, AgentFinish):
            raise ValueError("Only AgentAction with tool call may be converted to pydantic object")
        return self.pydantic_class(**tool_call_actions[0].tool_input)

    def parse(self, text: str) -> AgentFinish:
        raise ValueError("Can only parse messages")
