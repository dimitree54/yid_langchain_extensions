import unittest

from langchain import hub
from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel as BaseModelV2, Field as FieldV2

from yid_langchain_extensions.llm.tools_llm_with_thought import build_tools_llm_with_thought


class PowerFnArgsV2(BaseModelV2):
    power: int = FieldV2(description="power")
    base: float = FieldV2(description="base")


class Reasoning(BaseModelV2):
    reasoning: str


class TestToolsLLMWithThought(unittest.TestCase):
    def test(self):
        tools = [convert_to_openai_tool(PowerFnArgsV2)]
        base_prompt = hub.pull("hwchase17/openai-tools-agent")
        thought_prompt: ChatPromptTemplate = hub.pull("dimitree54/introduce_thought_tool")
        llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)  # noqa
        llm_with_thought = build_tools_llm_with_thought(llm, tools, thought_prompt, Reasoning)
        extended_chain = base_prompt | llm_with_thought | PydanticToolsParser(tools=[PowerFnArgsV2])
        result = extended_chain.invoke({"input": "3.43^5", "agent_scratchpad": [], "chat_history": []})
        self.assertEqual(result[0].base, 3.43)
        self.assertEqual(result[0].power, 5)
