import copy
from typing import Optional, Any

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel, LanguageModelInput, LanguageModelOutput
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input, Output

ASK_TO_REFORMAT_PROMPT = PromptTemplate.from_template(
    "Thank you for your answer, but it does not follow the output formatting instructions. "
    "The following error occurred:\nOutput parsing error:\n{error_message}\n\n"
    "Right now you HAVE TO *RE-FORMAT* your answer to meet the requirements and to fix the error!!!. "
    "NOTE: NO NEED to re-do the job, just re-format it properly!!!"
)


class LLMWithParsingRetry(Runnable[LanguageModelInput, Any]):
    def __init__(
            self, llm: BaseChatModel, parser: Runnable[LanguageModelOutput, Any],
            max_retries: int = 1, exceptions_to_retry: tuple[type[Exception]] = (OutputParserException,),
            reformat_prompt: PromptTemplate = ASK_TO_REFORMAT_PROMPT,
    ):
        self.llm = llm
        self.parser = parser
        self.max_retries = max_retries
        self.exceptions_to_retry = exceptions_to_retry
        self.reformat_prompt = reformat_prompt

    def _extend_input(
            self, input: LanguageModelInput, bad_result: LanguageModelOutput, error_message: str
    ) -> LanguageModelInput:
        error_message = self.reformat_prompt.format(error_message=error_message)
        if isinstance(input, str):
            bad_result_str = bad_result if isinstance(bad_result, str) else bad_result.content
            return input + "\n\n" + f"AI: {bad_result_str}" + "\n\n" + error_message
        bad_result_message = bad_result if isinstance(bad_result, AIMessage) else AIMessage(content=bad_result.content)

        error_messages = []
        if bad_result_message.tool_calls:
            # some llms require all tool calls to have a result:
            for tool_call in bad_result_message.tool_calls:
                error_messages.append(
                    ToolMessage(
                        tool_call_id=tool_call["id"],
                        content="tool call failed because of parsing error (check further)")
                )
        error_messages.append(HumanMessage(content=error_message))
        if isinstance(input, list):
            return input + [bad_result_message] + error_messages
        elif isinstance(input, ChatPromptValue):
            input_copy = copy.deepcopy(input)
            input_copy.messages.append(bad_result_message)
            input_copy.messages.extend(error_messages)
            return input_copy
        raise NotImplementedError(f"type {type(input)} not supported")

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Any:
        aggregated_error = ""
        extended_input = input
        for _ in range(self.max_retries + 1):
            llm_output = self.llm.invoke(extended_input, config, **kwargs)
            try:
                parser_output = self.parser.invoke(llm_output, config, **kwargs)
                return parser_output
            except self.exceptions_to_retry as e:
                aggregated_error += str(e) + "\n\n"
                extended_input = self._extend_input(extended_input, llm_output, str(e))
        raise OutputParserException(f"Failed to parse LLM output after {self.max_retries} retries:\n{aggregated_error}")

    async def ainvoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        aggregated_error = ""
        extended_input = input
        for _ in range(self.max_retries + 1):
            llm_output = await self.llm.ainvoke(extended_input, config, **kwargs)
            try:
                parser_output = await self.parser.ainvoke(llm_output, config, **kwargs)
                return parser_output
            except self.exceptions_to_retry as e:
                aggregated_error += str(e) + "\n\n"
                extended_input = self._extend_input(extended_input, llm_output, str(e))
        raise OutputParserException(f"Failed to parse LLM output after {self.max_retries} retries:\n{aggregated_error}")
