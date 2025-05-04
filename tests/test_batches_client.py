import unittest
from typing import Tuple

import numpy as np
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from yid_langchain_extensions.llm.batches_openai_client import BatchesOpenAICompletions
from yid_langchain_extensions.utils import encode_image_to_url


class TestBatchesClient(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        batches_client = BatchesOpenAICompletions()
        self.llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0, async_client=batches_client)

    @staticmethod
    def generate_random_image():
        random_image_array = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        return random_image_array

    async def test_chain_with_text(self) -> Tuple[bool, str]:
        """Test the batches client with text input in a chain."""
        try:
            prompt = ChatPromptTemplate.from_messages([HumanMessage(content="{message}")])
            chain = prompt | self.llm | StrOutputParser()
            answer = await chain.ainvoke(input={"message": "hi"})
            return len(answer) > 0, "Text chain test passed"
        except Exception as e:
            return False, f"Text chain test failed: {str(e)}"

    async def test_chain_with_inline_image(self) -> Tuple[bool, str]:
        """Test the batches client with an inline image."""
        try:
            chain = self.llm | StrOutputParser()

            content = [
                {'type': 'text', 'text': 'hi'},
                {'type': 'image_url', 'image_url': {
                    'url': encode_image_to_url(self.generate_random_image())
                }}
            ]

            answer = await chain.ainvoke(input=[HumanMessage(content=content)])
            return len(answer) > 0, "Inline image test passed"
        except Exception as e:
            return False, f"Inline image test failed: {str(e)}"

    async def test_chain_with_online_image(self) -> Tuple[bool, str]:
        """Test the batches client with an online image."""
        try:
            chain = self.llm | StrOutputParser()

            url = ("https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/"
                   "Gfp-wisconsin-madison-the-nature-boardwalk.jpg/"
                   "2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg")

            content = [
                {'type': 'text', 'text': 'hi'},
                {'type': 'image_url', 'image_url': {
                    'url': url
                }}
            ]

            answer = await chain.ainvoke(input=[HumanMessage(content=content)])
            return len(answer) > 0, "Online image test passed"
        except Exception as e:
            return False, f"Online image test failed: {str(e)}"
