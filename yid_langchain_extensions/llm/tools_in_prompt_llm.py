import copy
import secrets
import string
import typing
from typing import Sequence, Union, Any, Callable, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import LanguageModelInput, BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, ToolCall, HumanMessage
from langchain_core.output_parsers import JsonOutputParser, BaseCumulativeTransformOutputParser
from langchain_core.outputs import ChatResult
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field


def add_tool_calls(base_input: LanguageModelInput, extra_message: str) -> LanguageModelInput:
    if isinstance(base_input, str):
        return [HumanMessage(content=base_input), SystemMessage(content=extra_message)]
    if isinstance(base_input, list):
        return base_input + [SystemMessage(content=extra_message)]
    if isinstance(base_input, ChatPromptValue):
        base_input_copy = copy.deepcopy(base_input)
        base_input_copy.messages.append(SystemMessage(content=extra_message))
        return base_input_copy
    raise NotImplementedError(f"type {type(base_input)} not supported")


def split_thinking_and_output(text: str) -> (str, str):
    start_tag = "<think>"
    end_tag = "</think>"
    start = text.find(start_tag)
    if start == -1:
        return "", text
    end = text.find(end_tag, start)
    if end == -1:
        thoughts = text[start + len(start_tag):]
        return thoughts, ""
    thoughts = text[start + len(start_tag):end]
    output = text[end + len(end_tag):]
    return thoughts.strip(), output.strip()


def generate_call_id(length: int = 24, prefix: str = "call_") -> str:
    alphabet = string.ascii_letters + string.digits
    rand_part = ''.join(secrets.choice(alphabet) for _ in range(length))
    return prefix + rand_part


class DeepseekR1JsonToolCallsParser(BaseCumulativeTransformOutputParser[Any]):
    base_json_parser: JsonOutputParser = Field(default_factory=JsonOutputParser)
    raise_if_cannot_parse: bool = False

    def parse(self, text: str) -> AIMessage:
        thoughts, output = split_thinking_and_output(text)
        try:
            raw_tool_calls = self.base_json_parser.parse(output)
        except OutputParserException as e:
            if self.raise_if_cannot_parse:
                raise e
            else:
                return AIMessage(content=output, additional_kwargs={"thoughts": thoughts, "parsing_error": str(e)})
        if isinstance(raw_tool_calls, dict):
            raw_tool_calls = [raw_tool_calls]

        tool_calls = []
        try:
            for raw_tool_call in raw_tool_calls:
                tool_calls.append(
                    ToolCall(
                        name=raw_tool_call["name"],
                        args=raw_tool_call["arguments"],
                        id=generate_call_id()
                    )
                )
        except KeyError as e:
            if self.raise_if_cannot_parse:
                raise OutputParserException(f"Expected key not found in output json: {str(e)}")
            else:
                return AIMessage(content=output, additional_kwargs={"thoughts": thoughts, "parsing_error": str(e)})
        return AIMessage(content="", additional_kwargs={"thoughts": thoughts}, tool_calls=tool_calls)


class ModelWithPromptIntroducedTools(BaseChatModel):
    """
    Some models do not support tools calling out-of-the box,
     but they are still smart enough to properly follow the schema.
    So you can use this wrapper for such models, to support binding tools for them.
    Tools will be introduced as part of the input prompt.
    """
    base_model: BaseChatModel

    @classmethod
    def wrap_model(cls, base_model: BaseChatModel) -> "ModelWithPromptIntroducedTools":
        return ModelWithPromptIntroducedTools(
            base_model=base_model,
            name=base_model.name,
            cache=base_model.cache,
            verbose=base_model.verbose,
            callbacks=base_model.callbacks,
            tags=base_model.tags,
            metadata=base_model.metadata,
            custom_get_token_ids=base_model.custom_get_token_ids,
            callback_manager=base_model.callback_manager,
            rate_limiter=base_model.rate_limiter,
            disable_streaming=base_model.disable_streaming,
        )

    def _generate(self, messages: list[BaseMessage], stop: Optional[list[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> ChatResult:
        return self.base_model._generate(messages, stop, run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return self.base_model._llm_type

    def bind_tools(
            self,
            tools: Sequence[Union[dict[str, Any], type, Callable, BaseTool]],
            *,
            tool_choice: Optional[
                Union[dict, str, typing.Literal["auto", "none", "required", "any"], bool]
            ] = None,
            strict: Optional[bool] = None,
            parallel_tool_calls: Optional[bool] = None,
            **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        formatted_tools = [
            convert_to_openai_tool(tool, strict=strict) for tool in tools
        ]
        formatted_tools = {
            tool["function"]["name"]: tool for tool in formatted_tools
        }

        match tool_choice:
            case "none":
                return self
            case "any" | "required":
                if parallel_tool_calls:
                    suffix = ("Right now you should call one or several tools from this list. "
                              "It is forbidden to answer with plain text right now. "
                              "Return a list of tool calls (it might be a length of 1)")
                else:
                    suffix = ("Right now you should call one tool from this list. "
                              "It is forbidden to answer with plain text right now. "
                              "Return a tool call json.")
            case "auto" | None | False:
                if parallel_tool_calls:
                    suffix = ("Right now you should call one or several tools from this list. "
                              "Or you can answer with plain text to user instead of calling tools. "
                              "Return a plain text or a list of tool calls (it might be a length of 1)")
                else:
                    suffix = ("Right now you should call some tool from this list. "
                              "Or you can answer with plain text to user instead of calling any tool. "
                              "Return a plain text or a tool call json.")
            case cmd if cmd in formatted_tools.keys():
                formatted_tools = {k: v for k, v in formatted_tools.items() if k == tool_choice}
                suffix = (f"Right now you should call tool '{tool_choice}'. "
                          "Return a tool call json.")
            case _:
                raise ValueError("Unsupported value of tool_choice argument")

        tools_intro = f"You have access to {len(formatted_tools)} tools with following schemas:\n"
        for tool_name, formatted_tool in formatted_tools.items():
            tools_intro += f"{formatted_tool}\n"
        tools_intro += (f"\n{suffix}\nHint: before actually calling the tool,"
                        f" think well, how are you going to call it. "
                        f"During thinking, make sure you precisely follow the tool schema!!! ")

        if parallel_tool_calls:
            tools_intro += """
Example of tools calling:
```json
[
    {
        "name": "tool1_name",
        "arguments": {
            "int_arg_name": int_value,
            "bool_arg_name": true,
        }
    },
    {
        "name": "tool2_name",
        "arguments": {
            "optional_arg_name": null,
            "str_arg_name": "str_value"
        }
    },
]
```
"""
        else:
            tools_intro += """
Example of tool calling:
```json
{
    "name": "tool_name",
    "arguments": {
        "int_arg_name": int_value,
        "bool_arg_name": true,
        "optional_arg_name": null,
        "str_arg_name": "str_value"
    }
}
```
"""

        return (lambda input: add_tool_calls(input, tools_intro)) | self
