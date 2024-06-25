import asyncio
import json
import tempfile
import uuid
from typing import Iterable, Union, Optional, Dict, List

import httpx
from openai import NotGiven, NOT_GIVEN, AsyncOpenAI
from openai._types import Headers, Query, Body
from openai._utils import maybe_transform, required_args
from openai.resources.chat import AsyncCompletions
from openai.types import ChatModel
from openai.types.chat import ChatCompletionMessageParam, completion_create_params, ChatCompletionStreamOptionsParam, \
    ChatCompletionToolChoiceOptionParam, ChatCompletionToolParam, ChatCompletion
from typing_extensions import Literal


class BatchesOpenAICompletions(AsyncCompletions):
    def __init__(self, client: AsyncOpenAI | None = None):
        super().__init__(client=client or AsyncOpenAI())

    def filter_not_given(self, params: Dict[str, str]) -> Dict[str, str]:
        return {key: value for key, value in params.items() if value != NOT_GIVEN}

    @required_args(["messages", "model"], ["messages", "model", "stream"])
    async def create(
        self,
        *,
        messages: Iterable[ChatCompletionMessageParam],
        model: Union[str, ChatModel],
        frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        function_call: completion_create_params.FunctionCall | NotGiven = NOT_GIVEN,
        functions: Iterable[completion_create_params.Function] | NotGiven = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
        logprobs: Optional[bool] | NotGiven = NOT_GIVEN,
        max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        n: Optional[int] | NotGiven = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        response_format: completion_create_params.ResponseFormat | NotGiven = NOT_GIVEN,
        seed: Optional[int] | NotGiven = NOT_GIVEN,
        stop: Union[Optional[str], List[str]] | NotGiven = NOT_GIVEN,
        stream: Optional[Literal[False]] | Literal[True] | NotGiven = NOT_GIVEN,
        stream_options: Optional[ChatCompletionStreamOptionsParam] | NotGiven = NOT_GIVEN,
        temperature: Optional[float] | NotGiven = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        tools: Iterable[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        top_logprobs: Optional[int] | NotGiven = NOT_GIVEN,
        top_p: Optional[float] | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        # this params are ignored
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> ChatCompletion:
        job_id = str(uuid.uuid4())
        body = maybe_transform(
            {
                "messages": messages,
                "model": model,
                "frequency_penalty": frequency_penalty,
                "function_call": function_call,
                "functions": functions,
                "logit_bias": logit_bias,
                "logprobs": logprobs,
                "max_tokens": max_tokens,
                "n": n,
                "parallel_tool_calls": parallel_tool_calls,
                "presence_penalty": presence_penalty,
                "response_format": response_format,
                "seed": seed,
                "stop": stop,
                "stream": False,
                "stream_options": NOT_GIVEN,
                "temperature": temperature,
                "tool_choice": tool_choice,
                "tools": tools,
                "top_logprobs": top_logprobs,
                "top_p": top_p,
                "user": user,
            },
            completion_create_params.CompletionCreateParams,
        )
        request = {
            "body": self.filter_not_given(body),
            "custom_id": job_id,
            "method": "POST",
            "url": "/v1/chat/completions",
        }
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w") as input_file:
            # writing to input file
            input_file.write(json.dumps(request))
            input_file.flush()

            # uploading input file
            with open(input_file.name, "rb") as f:
                batch_input_file = await self._client.files.create(
                    file=f,
                    purpose="batch"
                )

        # launching
        batch_launch_info = await self._client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": job_id}
        )

        # waiting to finish
        while True:
            await asyncio.sleep(5)
            batch_updated_info = await self._client.batches.retrieve(batch_launch_info.id)
            if batch_updated_info.status in ["validating", "in_progress", "finalizing", "cancelling"]:
                continue  # still busy
            elif batch_updated_info.status == "completed" and batch_updated_info.output_file_id:  # done
                # getting result
                result_content = await self._client.files.content(batch_updated_info.output_file_id)
                # parsing result
                json_string = result_content.content.decode(result_content.encoding)
                data_dict = json.loads(json_string)
                completion = ChatCompletion(**data_dict["response"]["body"])

                # cleanup files
                await self._client.files.delete(batch_input_file.id)
                await self._client.files.delete(batch_updated_info.output_file_id)

                return completion
            else:  # batch failed
                await self._client.files.delete(batch_input_file.id)
                if batch_updated_info.error_file_id:
                    result_content = await self._client.files.content(batch_updated_info.error_file_id)
                    json_string = result_content.content.decode(result_content.encoding)
                    data_dict = json.loads(json_string)
                    await self._client.files.delete(batch_updated_info.error_file_id)
                    raise Exception(data_dict["response"]["body"])
                else:
                    raise Exception("LLM failed with unknown reason")
