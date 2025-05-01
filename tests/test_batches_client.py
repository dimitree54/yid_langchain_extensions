import unittest

import numpy as np
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from yid_langchain_extensions.llm.batches_openai_client import BatchesOpenAICompletions
from yid_langchain_extensions.utils import encode_image_to_url


class TestBatchesClient(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        batches_client = BatchesOpenAICompletions()
        self.llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0, async_client=batches_client)

    async def test_batches_client_in_chain(self):
        prompt = ChatPromptTemplate.from_messages([HumanMessage(content="{message}")])
        chain = prompt | self.llm | StrOutputParser()
        answer = await chain.ainvoke(input={"message": "hi"})
        self.assertTrue(len(answer) > 0)

    def generate_random_image(self):
        random_image_array = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        return random_image_array

    async def test_batches_client_with_inline_image(self):
        chain = self.llm | StrOutputParser()

        content = [
            {'type': 'text', 'text': 'hi'},
            {'type': 'image_url', 'image_url': {
                'url': encode_image_to_url(self.generate_random_image())
            }}
        ]

        answer = await chain.ainvoke(input=[HumanMessage(content=content)])
        self.assertTrue(len(answer) > 0)

    async def test_batches_client_with_online_image(self):
        chain = self.llm | StrOutputParser()

        url = ("https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/"
               "Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg")

        content = [
            {'type': 'text', 'text': 'hi'},
            {'type': 'image_url', 'image_url': {
                'url': url
            }}
        ]

        answer = await chain.ainvoke(input=[HumanMessage(content=content)])
        self.assertTrue(len(answer) > 0)
