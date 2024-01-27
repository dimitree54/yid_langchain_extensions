from unittest import TestCase

from langchain_community.llms.fake import FakeListLLM
from langchain_core.language_models import LLM
from pydantic import BaseModel

from yid_langchain_extensions.utils import pydantic_v1_port


class Pydantic2ClassWithoutPort(BaseModel):
    llm: LLM


class Pydantic2ClassWithPort(BaseModel):
    llm: pydantic_v1_port(LLM)


class TestPydanticV1Port(TestCase):
    def setUp(self):
        self.llm = FakeListLLM(responses=[])

    def test_v1_port(self):
        Pydantic2ClassWithPort(llm=self.llm)

    def test_port_required(self):
        with self.assertRaises(TypeError):
            Pydantic2ClassWithoutPort(llm=self.llm)
