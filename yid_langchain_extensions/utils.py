import base64
from abc import ABC, abstractmethod
from typing import Dict, Union, Optional, Sequence, List

import cv2
import numpy as np
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnableConfig, Runnable
from langchain_core.utils.function_calling import _rm_titles, convert_to_openai_function, convert_to_openai_tool  # noqa
from pydantic import BaseModel


class ChatPromptValue2DictAdapter(Runnable[Union[ChatPromptValue, Dict], Dict[str, Sequence[BaseMessage]]]):
    def invoke(
            self, input: Union[ChatPromptValue, Dict], config: Optional[RunnableConfig] = None
    ) -> Dict[str, Sequence[BaseMessage]]:
        if isinstance(input, Dict):
            return input
        return {
            "messages": input.messages
        }


# todo refactor it to be runnable
class ContextSizeLimiter(BaseModel, ABC):
    @abstractmethod
    def limit_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        pass


class NaiveContextSizeLimiter(ContextSizeLimiter):
    max_context_size: int
    llm: BaseChatModel

    def limit_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        num_messages_to_keep = 1
        while num_messages_to_keep <= len(messages):
            extended_num_messages_to_keep = num_messages_to_keep + 1
            last_messages = messages[-extended_num_messages_to_keep:]
            tokens_in_extended_messages = self.llm.get_num_tokens_from_messages(last_messages)
            if tokens_in_extended_messages > self.max_context_size:
                break
            num_messages_to_keep = extended_num_messages_to_keep
        return messages[-num_messages_to_keep:]


class FirstMessageAuthorContextSizeLimiter(ContextSizeLimiter):
    first_message_author: str
    base_limiter: ContextSizeLimiter

    def _remove_head_until_author(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        for i in range(len(messages)):
            if messages[i].type == self.first_message_author:
                return messages[i:]
        return []

    def limit_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        messages = self.base_limiter.limit_messages(messages)
        return self._remove_head_until_author(messages)


def encode_image_to_url(image: np.ndarray) -> str:
    # Ensure the image is in 3-channel format
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a 3-channel image")

    # Encode the image to PNG format
    _, buffer = cv2.imencode('.png', image)

    # Convert the buffer to base64
    base64_image = base64.b64encode(buffer).decode('utf-8')

    # Create the data URL
    base64_url = f"data:image/png;base64,{base64_image}"

    return base64_url
