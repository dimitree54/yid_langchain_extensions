import unittest

from langchain import hub
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel as BaseModelV2, Field as FieldV2

from yid_langchain_extensions.llm.tools_llm_with_thought import build_tools_llm_with_thought
from yid_langchain_extensions.output_parser.pydantic_from_tool import PydanticOutputParser
from yid_langchain_extensions.utils import convert_to_openai_tool_v2


class PowerFnArgsV2(BaseModelV2):
    power: int = FieldV2(description="power")
    base: float = FieldV2(description="base")


class Reasoning(BaseModelV2):
    reasoning: str


class TestToolsLLMWithThought(unittest.TestCase):
    def test(self):
        tools = [
            convert_to_openai_tool_v2(PowerFnArgsV2)
        ]
        base_prompt = hub.pull("hwchase17/openai-tools-agent")
        thought_prompt: ChatPromptTemplate = hub.pull("dimitree54/introduce_thought_tool")
        llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)  # noqa
        llm_with_thought = build_tools_llm_with_thought(llm, tools, thought_prompt, Reasoning)
        extended_chain = base_prompt | llm_with_thought | PydanticOutputParser(
            pydantic_class=PowerFnArgsV2, base_parser=OpenAIToolsAgentOutputParser())
        result = extended_chain.invoke({"input": "3.43^5", "agent_scratchpad": [], "chat_history": []})
        self.assertEqual(result.base, 3.43)
        self.assertEqual(result.power, 5)
