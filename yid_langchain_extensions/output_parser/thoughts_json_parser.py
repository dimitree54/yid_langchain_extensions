import json
from typing import List, Dict, Any

from langchain import PromptTemplate
from langchain.schema import BaseOutputParser
from pydantic import BaseModel, validator

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

    @validator("thoughts")
    def validate_thoughts(cls, v):
        assert len(v) > 0, "You must have at least one thought"
        return v

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
