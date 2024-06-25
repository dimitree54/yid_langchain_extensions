import unittest

import numpy as np
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from yid_langchain_extensions.llm.batches_openai_client import BatchesOpenAICompletions
from yid_langchain_extensions.utils import encode_image_to_url


class TestBatchesClient(unittest.IsolatedAsyncioTestCase):
    async def test_batches_client_in_chain(self):
        batches_client = BatchesOpenAICompletions()
        llm = ChatOpenAI(async_client=batches_client)
        chain = llm | StrOutputParser()
        answer = await chain.ainvoke(input=[HumanMessage(content="hi")])
        self.assertTrue(len(answer) > 0)

    def generate_random_image(self):
        random_image_array = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        return random_image_array

    async def test_batches_client_with_inline_image(self):
        batches_client = BatchesOpenAICompletions()
        llm = ChatOpenAI(model_name="gpt-4o", async_client=batches_client)
        chain = llm | StrOutputParser()

        content = [
            {'type': 'text', 'text': 'hi'},
            {'type': 'image_url', 'image_url': {
                'url': encode_image_to_url(self.generate_random_image())
            }}
        ]

        answer = await chain.ainvoke(input=[HumanMessage(content=content)])
        self.assertTrue(len(answer) > 0)

    async def test_batches_client_with_online_image(self):
        batches_client = BatchesOpenAICompletions()
        llm = ChatOpenAI(model_name="gpt-4o", async_client=batches_client)
        chain = llm | StrOutputParser()

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
