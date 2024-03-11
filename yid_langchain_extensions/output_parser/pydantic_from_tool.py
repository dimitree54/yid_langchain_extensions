from typing import Union, Type

from langchain.agents.agent import MultiActionAgentOutputParser
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.agents import AgentFinish
from langchain_core.messages import AIMessage
from pydantic import BaseModel as BaseModelV2
from pydantic.v1 import BaseModel as BaseModelV1


class PydanticOutputParser(MultiActionAgentOutputParser):
    pydantic_class: Union[Type[BaseModelV1], Type[BaseModelV2]]
    base_parser: MultiActionAgentOutputParser = OpenAIToolsAgentOutputParser()

    @property
    def _type(self) -> str:
        return f"pydantic-{self.base_parser._type}"

    def parse_result(
            self, result: AIMessage, *, partial: bool = False
    ) -> Union[BaseModelV1, BaseModelV2]:
        tool_call_actions = self.base_parser.parse_result(result, partial=partial)
        if isinstance(tool_call_actions, AgentFinish):
            raise ValueError("Only AgentAction with tool call may be converted to pydantic object")
        return self.pydantic_class(**tool_call_actions[0].tool_input)

    def parse(self, text: str) -> AgentFinish:
        raise ValueError("Can only parse messages")
