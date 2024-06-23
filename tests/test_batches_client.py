import unittest

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from yid_langchain_extensions.llm.batches_openai_client import BatchesOpenAICompletions


class TestBatchesClient(unittest.IsolatedAsyncioTestCase):
    async def test_batches_client(self):
        batches_client = BatchesOpenAICompletions()
        llm = ChatOpenAI(async_client=batches_client)
        answer = await llm.agenerate(messages=[[HumanMessage(content="hi")]])
        print(answer)
