import unittest

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from yid_langchain_extensions.llm.retrying_llm import LLMWithParsingRetry


class Dot(BaseModel):
    """Multiplies a and b"""
    a: int
    b: int


class TestRetryingLLM(unittest.TestCase):
    def setUp(self):
        self.llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)

    def test_with_chat_messages(self):
        retrying_llm = LLMWithParsingRetry(
            llm=self.llm,
            parser=PydanticOutputParser(pydantic_object=Dot),
            max_retries=3
        )

        answer: AIMessage = retrying_llm.invoke([HumanMessage(content="return json call of function Dot(2,7)")])
        self.assertEqual(answer, Dot(a=2, b=7))


class TestRetryingLLMAsync(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)

    async def test_with_chat_messages(self):
        retrying_llm = LLMWithParsingRetry(
            llm=self.llm,
            parser=PydanticOutputParser(pydantic_object=Dot),
            max_retries=3
        )

        answer: AIMessage = await retrying_llm.ainvoke([HumanMessage(content="return json call of function Dot(2,7)")])
        self.assertEqual(answer, Dot(a=2, b=7))
