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
