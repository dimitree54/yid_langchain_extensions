import json
from typing import List, Dict, Any

from langchain import PromptTemplate
from langchain.schema import AgentAction, BaseOutputParser
from pydantic import BaseModel

FORMAT_PROMPT = """FORMAT:
------
You response should be in the following format:

```json
{
    {{thoughts}}
}
```"""


class Thought(BaseModel):
    name: str
    description: str
    type: str = "string"


class ThoughtsJSONParser(BaseOutputParser):
    stop_sequences: List[str] = ["}\n```", "}```"]
    thoughts: List[Thought] = []

    def parse(self, text: str) -> Dict[str, Any]:
        text += self.stop_sequences[0]
        cleaned_output = text.strip()
        if "```json" in cleaned_output:
            _, cleaned_output = cleaned_output.split("```json")
        if "```" in cleaned_output:
            cleaned_output, _ = cleaned_output.split("```")
        if cleaned_output.startswith("```json"):
            cleaned_output = cleaned_output[len("```json"):]
        if cleaned_output.startswith("```"):
            cleaned_output = cleaned_output[len("```"):]
        if cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[: -len("```")]
        cleaned_output = cleaned_output.strip()
        response = json.loads(cleaned_output)
        return response

    def format_thoughts(self) -> str:
        return "\n\t".join([
            f'"{thought.name}": {thought.type} [{thought.description}]' for thought in self.thoughts
        ])

    def get_format_instructions(self) -> str:
        format_instructions = PromptTemplate.from_template(FORMAT_PROMPT, template_format="jinja2").format_prompt(
            thoughts=self.format_thoughts()).to_string()
        return format_instructions


class ActionParser(ThoughtsJSONParser):
    extra_thoughts: List[Thought] = []
    action_thoughts: List[Thought] = [
        Thought(name="action", description="The action to take. Must be one of [{tool_names}]"),
        Thought(name="action_input", description="The input to the action")
    ]

    @property
    def thoughts(self) -> List[Thought]:
        return self.action_thoughts + self.extra_thoughts

    def parse(self, text: str) -> AgentAction:
        response = super().parse(text)
        return AgentAction(response["action"], response["action_input"], text)
