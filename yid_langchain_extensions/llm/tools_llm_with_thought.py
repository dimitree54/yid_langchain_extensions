import copy
from typing import Dict, Type, Optional, Any, List, AsyncIterator

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnablePassthrough
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from yid_langchain_extensions.utils import ChatPromptValue2DictAdapter


class ThoughtStripper(Runnable[AIMessageChunk, AIMessageChunk]):
    def __init__(self, thought_name: str):
        self.thought_name = thought_name
        self.silenced = False

    def invoke(
            self,
            input: AIMessage,  # noqa
            config: Optional[RunnableConfig] = None
    ) -> AIMessage:
        return self._strip(input)

    async def atransform(
        self,
        input: AsyncIterator[AIMessageChunk],  # noqa
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[AIMessageChunk]:
        async for chunk in input:
            yield self._strip_from_chunk(chunk)
        self.silenced = False

    def _strip(self, message: AIMessage) -> AIMessage:
        message = copy.deepcopy(message)
        message.additional_kwargs["tool_calls"] = [tool_call for tool_call in message.additional_kwargs["tool_calls"]
                                                   if tool_call["function"]["name"] != self.thought_name]
        message.tool_calls = [tool_call for tool_call in message.tool_calls
                              if tool_call["name"] != self.thought_name]
        if isinstance(message, AIMessageChunk):
            message.tool_call_chunks = [tool_call for tool_call in message.tool_call_chunks
                                        if tool_call["name"] != self.thought_name]
        return message

    def _strip_from_chunk(self, message: AIMessageChunk) -> AIMessageChunk:
        message = copy.deepcopy(message)
        for tool_call_chunk in message.tool_call_chunks:
            tool_name = tool_call_chunk["name"]
            if tool_name == self.thought_name:
                self.silenced = True
            elif tool_name is not None:
                self.silenced = False
        if self.silenced:
            message.additional_kwargs["tool_calls"] = []
            message.tool_calls = []
            message.tool_call_chunks = []
        return message


def build_tools_llm_with_thought(
        tools_llm: ChatOpenAI,
        openai_tools: List[Dict[str, Any]],
        thought_introducing_prompt: ChatPromptTemplate,
        thought_class: Type[BaseModel]
) -> Runnable[ChatPromptValue, AIMessage]:
    thought_tool = convert_to_openai_tool(thought_class)
    all_tools = [thought_tool] + openai_tools
    llm_with_thought_tool = tools_llm.bind(tools=all_tools)

    thought_tool_name = thought_tool["function"]["name"]
    thought_stripper = ThoughtStripper(thought_name=thought_tool_name)
    return ChatPromptValue2DictAdapter() | RunnablePassthrough.assign(
        thought_tool_name=lambda x: thought_tool_name
    ) | thought_introducing_prompt | llm_with_thought_tool | thought_stripper
