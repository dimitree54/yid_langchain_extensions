import unittest

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from yid_langchain_extensions.llm.tools_in_prompt_llm import (
    DeepseekR1JsonToolCallsParser, ModelWithPromptIntroducedTools)


@tool
def add(a, b):
    """Adds a and b"""
    return a + b


class Dot(BaseModel):
    """Multiplies a and b"""
    a: int
    b: int


class TestModelWithTools(unittest.TestCase):
    def setUp(self):
        # actually o4-mini supports tools out of the box, but it is easiest to set up as an example
        test_llm = ChatOpenAI(model_name="o4-mini-2025-04-16", model_kwargs={"service_tier": "flex"})
        self.llm = ModelWithPromptIntroducedTools.wrap_model(base_model=test_llm)

    def test_with_chat_messages(self):
        chain_sequential = self.llm.bind_tools(
            tools=[add, Dot], tool_choice="auto") | DeepseekR1JsonToolCallsParser()

        answer: AIMessage = chain_sequential.invoke([HumanMessage(content="hi")])
        self.assertEqual(len(answer.tool_calls), 0)

    def test_with_prompt_template(self):
        prompt = ChatPromptTemplate.from_messages([HumanMessage(content="{message}")])
        chain_sequential = prompt | self.llm.bind_tools(
            tools=[add, Dot], tool_choice="auto") | DeepseekR1JsonToolCallsParser()
        answer: AIMessage = chain_sequential.invoke(input={"message": "hi"})
        self.assertEqual(len(answer.tool_calls), 0)

    def test_auto_sequential_tool_choice(self):
        chain_sequential = self.llm.bind_tools(
            tools=[add, Dot], tool_choice="auto") | DeepseekR1JsonToolCallsParser()

        answer: AIMessage = chain_sequential.invoke("hi")
        self.assertEqual(len(answer.tool_calls), 0)

        answer: AIMessage = chain_sequential.invoke("call tool add(3,5)")
        self.assertEqual(len(answer.tool_calls), 1)
        self.assertEqual(answer.tool_calls[0]["name"], "add")
        self.assertEqual(answer.tool_calls[0]["args"].get("a"), 3)
        self.assertEqual(answer.tool_calls[0]["args"].get("b"), 5)
        self.assertIsNotNone(answer.tool_calls[0]["id"])

    def test_auto_sequential_tool_choice_tricky(self):
        chain_sequential = self.llm.bind_tools(
            tools=[add, Dot], tool_choice="auto") | DeepseekR1JsonToolCallsParser()

        answer: AIMessage = chain_sequential.invoke("call tool add(3,5) and Dot(2,7)")
        self.assertEqual(len(answer.tool_calls), 1)

    def test_auto_parallel_tool_choice(self):
        chain_parallel = self.llm.bind_tools(
            tools=[add, Dot], tool_choice="auto", parallel_tool_calls=True) | DeepseekR1JsonToolCallsParser()

        answer: AIMessage = chain_parallel.invoke("hi")
        self.assertEqual(len(answer.tool_calls), 0)

        answer: AIMessage = chain_parallel.invoke("call tool add(3,5)")
        self.assertEqual(len(answer.tool_calls), 1)
        self.assertEqual(answer.tool_calls[0]["name"], "add")
        self.assertEqual(answer.tool_calls[0]["args"].get("a"), 3)
        self.assertEqual(answer.tool_calls[0]["args"].get("b"), 5)
        self.assertIsNotNone(answer.tool_calls[0]["id"])

        answer: AIMessage = chain_parallel.invoke("call tool add(3,5) and Dot(2,7)")
        self.assertEqual(len(answer.tool_calls), 2)
        self.assertEqual(answer.tool_calls[0]["name"], "add")
        self.assertEqual(answer.tool_calls[0]["args"].get("a"), 3)
        self.assertEqual(answer.tool_calls[0]["args"].get("b"), 5)
        self.assertIsNotNone(answer.tool_calls[0]["id"])
        self.assertEqual(len(answer.tool_calls), 2)
        self.assertEqual(answer.tool_calls[1]["name"], "Dot")
        self.assertEqual(answer.tool_calls[1]["args"].get("a"), 2)
        self.assertEqual(answer.tool_calls[1]["args"].get("b"), 7)
        self.assertIsNotNone(answer.tool_calls[1]["id"])

    def test_any_sequential_tool_choice(self):
        chain_sequential = self.llm.bind_tools(tools=[add, Dot], tool_choice="any") | DeepseekR1JsonToolCallsParser()

        answer: AIMessage = chain_sequential.invoke("hi")
        self.assertEqual(len(answer.tool_calls), 1)
        self.assertEqual(answer.tool_calls[0]["name"], "add")

        answer: AIMessage = chain_sequential.invoke("call tool add(3,5)")
        self.assertEqual(len(answer.tool_calls), 1)
        self.assertEqual(answer.tool_calls[0]["name"], "add")
        self.assertEqual(answer.tool_calls[0]["args"].get("a"), 3)
        self.assertEqual(answer.tool_calls[0]["args"].get("b"), 5)
        self.assertIsNotNone(answer.tool_calls[0]["id"])

        answer: AIMessage = chain_sequential.invoke("call tool add(3,5) and Dot(2,7)")
        self.assertEqual(len(answer.tool_calls), 1)

    def test_any_parallel_tool_choice(self):
        chain_parallel = self.llm.bind_tools(
            tools=[add, Dot], tool_choice="any", parallel_tool_calls=True) | DeepseekR1JsonToolCallsParser()

        answer: AIMessage = chain_parallel.invoke("call tool add(3,5)")
        self.assertEqual(len(answer.tool_calls), 1)
        self.assertEqual(answer.tool_calls[0]["name"], "add")
        self.assertEqual(answer.tool_calls[0]["args"].get("a"), 3)
        self.assertEqual(answer.tool_calls[0]["args"].get("b"), 5)
        self.assertIsNotNone(answer.tool_calls[0]["id"])

        answer: AIMessage = chain_parallel.invoke("call tool add(3,5) and Dot(2,7)")
        self.assertEqual(len(answer.tool_calls), 2)
        self.assertEqual(answer.tool_calls[0]["name"], "add")
        self.assertEqual(answer.tool_calls[0]["args"].get("a"), 3)
        self.assertEqual(answer.tool_calls[0]["args"].get("b"), 5)
        self.assertIsNotNone(answer.tool_calls[0]["id"])
        self.assertEqual(len(answer.tool_calls), 2)
        self.assertEqual(answer.tool_calls[1]["name"], "Dot")
        self.assertEqual(answer.tool_calls[1]["args"].get("a"), 2)
        self.assertEqual(answer.tool_calls[1]["args"].get("b"), 7)
        self.assertIsNotNone(answer.tool_calls[1]["id"])

    def test_any_parallel_tricky(self):
        chain_parallel = self.llm.bind_tools(
            tools=[add, Dot], tool_choice="any", parallel_tool_calls=True) | DeepseekR1JsonToolCallsParser()

        answer: AIMessage = chain_parallel.invoke("hi")
        self.assertGreater(len(answer.tool_calls), 0)
        for tool_call in answer.tool_calls:
            self.assertIn(tool_call["name"], {"add", "Dot"})

    def test_specific_sequential_tool_choice(self):
        chain_sequential = self.llm.bind_tools(tools=[add, Dot], tool_choice="add") | DeepseekR1JsonToolCallsParser()

        answer: AIMessage = chain_sequential.invoke("hi")
        self.assertEqual(len(answer.tool_calls), 1)
        self.assertEqual(answer.tool_calls[0]["name"], "add")

        answer: AIMessage = chain_sequential.invoke("call tool add(3,5)")
        self.assertEqual(len(answer.tool_calls), 1)
        self.assertEqual(answer.tool_calls[0]["name"], "add")
        self.assertEqual(answer.tool_calls[0]["args"].get("a"), 3)
        self.assertEqual(answer.tool_calls[0]["args"].get("b"), 5)
        self.assertIsNotNone(answer.tool_calls[0]["id"])

        answer: AIMessage = chain_sequential.invoke("call tool add(3,5) and Dot(2,7)")
        self.assertEqual(len(answer.tool_calls), 1)

    def test_specific_parallel_tool_choice(self):
        chain_parallel = self.llm.bind_tools(
            tools=[add, Dot], tool_choice="add", parallel_tool_calls=True) | DeepseekR1JsonToolCallsParser()

        answer: AIMessage = chain_parallel.invoke("call tool add(3,5) and add(2,7)")
        self.assertEqual(len(answer.tool_calls), 2)
        self.assertEqual(answer.tool_calls[0]["name"], "add")
        self.assertEqual(answer.tool_calls[0]["args"].get("a"), 3)
        self.assertEqual(answer.tool_calls[0]["args"].get("b"), 5)
        self.assertIsNotNone(answer.tool_calls[0]["id"])
        self.assertEqual(len(answer.tool_calls), 2)
        self.assertEqual(answer.tool_calls[1]["name"], "add")
        self.assertEqual(answer.tool_calls[1]["args"].get("a"), 2)
        self.assertEqual(answer.tool_calls[1]["args"].get("b"), 7)
        self.assertIsNotNone(answer.tool_calls[1]["id"])

    def test_specific_parallel_tool_choice_tricky(self):
        chain_parallel = self.llm.bind_tools(
            tools=[add, Dot], tool_choice="add", parallel_tool_calls=True) | DeepseekR1JsonToolCallsParser()

        answer: AIMessage = chain_parallel.invoke("hi")
        self.assertGreater(len(answer.tool_calls), 0)
        for tool_call in answer.tool_calls:
            self.assertEqual(tool_call["name"], "add")
