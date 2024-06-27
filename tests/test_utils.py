from typing import List, Dict
from unittest import TestCase

from langchain_community.llms.fake import FakeListLLM
from langchain_core.language_models import LLM
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from yid_langchain_extensions.utils import pydantic_v1_port, FirstMessageAuthorContextSizeLimiter, \
    NaiveContextSizeLimiter


class Pydantic2ClassWithoutPort(BaseModel):
    llm: LLM


class Pydantic2ClassWithPort(BaseModel):
    llm: pydantic_v1_port(LLM)


class Pydantic2ClassWithPortNested(BaseModel):
    llm_list: pydantic_v1_port(List[LLM])
    llm_dict: pydantic_v1_port(Dict[str, LLM])
    complex_field: pydantic_v1_port(RunnableSerializable[List[BaseMessage], str])


class TestPydanticV1Port(TestCase):
    def setUp(self):
        self.llm = FakeListLLM(responses=[])

    def test_v1_port(self):
        Pydantic2ClassWithPort(llm=self.llm)

    def test_port_required(self):
        with self.assertRaises(TypeError):
            Pydantic2ClassWithoutPort(llm=self.llm)

    def test_v1_port_nested(self):
        Pydantic2ClassWithPortNested(
            llm_list=[self.llm],
            llm_dict={"llm": self.llm},
            complex_field=self.llm
        )


class TestFirstMessageAuthorContextSizeLimiter(TestCase):
    def test_limit_messages(self):
        messages = [
            HumanMessage(content="hi"),
            SystemMessage(content="say hi"),
            AIMessage(content="hi"),
            HumanMessage(content="how r u?"),
            SystemMessage(content="say how r u"),
            AIMessage(content="im fine"),
        ]
        limiter = FirstMessageAuthorContextSizeLimiter(
            first_message_author="human",
            base_limiter=NaiveContextSizeLimiter(
                max_context_size=40, llm=ChatOpenAI()
            )
        )
        limited_messages = limiter.limit_messages(messages)
        self.assertEqual(len(limited_messages), 3)
