from unittest import TestCase

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from yid_langchain_extensions.utils import FirstMessageAuthorContextSizeLimiter, \
    NaiveContextSizeLimiter


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
