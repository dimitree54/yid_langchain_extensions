from typing import List, Dict
from unittest import TestCase

from langchain_community.llms.fake import FakeListLLM
from langchain_core.language_models import LLM
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableSerializable
from pydantic import BaseModel

from yid_langchain_extensions.utils import pydantic_v1_port


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
