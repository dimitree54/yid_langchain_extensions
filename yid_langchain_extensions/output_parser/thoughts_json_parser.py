from typing import List, Dict, Any

from langchain import PromptTemplate
from langchain.output_parsers.json import parse_json_markdown
from langchain.schema import BaseOutputParser
from pydantic import BaseModel, validator

FORMAT_PROMPT = """RESPONSE FORMAT:
------
You response should be a markdown code snippet formatted in the following schema:

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
    thoughts: List[Thought]
    stop_sequences: List[str] = ["}\n```", "}```"]
    format_prompt: str = FORMAT_PROMPT

    @validator("thoughts")
    def validate_thoughts(cls, v):
        assert len(v) > 0, "You must have at least one thought"
        return v

    def parse(self, text: str) -> Dict[str, Any]:
        if text.startswith("```json"):
            text += self.stop_sequences[0]
        response = parse_json_markdown(text)
        return response

    def format_thoughts(self) -> str:
        return "\n\t".join([
            f'"{thought.name}": {thought.type} [{thought.description}]' for thought in self.thoughts
        ])

    def get_format_instructions(self) -> str:
        format_instructions = PromptTemplate.from_template(self.format_prompt, template_format="jinja2").format_prompt(
            thoughts=self.format_thoughts()).to_string()
        return format_instructions
