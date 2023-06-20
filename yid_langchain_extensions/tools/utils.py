from typing import List

from langchain import PromptTemplate
from langchain.tools import BaseTool

TOOLS_PROMPT = """AVAILABLE ACTIONS:
------
Now you have access to following actions in order to solve the task:
{tools}"""


def format_tools(tools: List[BaseTool]) -> str:
    tools_string = "\n".join(
        [f"> {tool.name}: {tool.description}" for tool in tools]
    )
    return PromptTemplate.from_template(TOOLS_PROMPT).format_prompt(
        tools=tools_string
    ).to_string()


def format_tool_names(tools: List[BaseTool]) -> str:
    return ", ".join([tool.name for tool in tools])


class FinalAnswerTool(BaseTool):
    name: str = "final_answer"
    description: str = "Use this if you want to respond directly to the human."
    return_direct: bool = True

    def _run(self, action_input: str) -> str:
        return action_input

    async def _arun(self, action_input: str) -> str:
        return action_input
