from typing import Union, Dict, Type, Optional

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnablePassthrough
from pydantic import BaseModel as BaseModelV2
from pydantic.v1 import BaseModel as BaseModelV1

from yid_langchain_extensions.utils import convert_to_openai_tool_v2, ListExtendingRunnableBinding


class ThoughtStripper(BaseModelV2, Runnable[AIMessage, AIMessage]):
    thought_name: str

    def invoke(self, input: AIMessage, config: Optional[RunnableConfig] = None) -> AIMessage:
        if "tool_calls" in input.additional_kwargs:
            input.additional_kwargs["tool_calls"] = [
                tool_call for tool_call in input.additional_kwargs["tool_calls"]
                if tool_call["function"]["name"] != self.thought_name
            ]
        return input


def build_tools_llm_with_thought(
        tools_llm: Runnable[Dict, AIMessage],
        thought_introducing_prompt: ChatPromptTemplate,
        thought_class: Union[Type[BaseModelV1], Type[BaseModelV2]]
) -> Runnable[Dict, AIMessage]:
    thought_tool = convert_to_openai_tool_v2(thought_class)
    llm_with_thought_tool = ListExtendingRunnableBinding(bound=tools_llm, kwargs={"tools": [thought_tool]})

    thought_tool_name = thought_tool["function"]["name"]
    thought_stripper = ThoughtStripper(thought_name=thought_tool_name)
    return RunnablePassthrough.assign(
        thought_tool_name=lambda x: thought_tool_name
    ) | thought_introducing_prompt | llm_with_thought_tool | thought_stripper
