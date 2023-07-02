import json
from typing import List, Dict, Any

from langchain import PromptTemplate
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


def escape_new_lines_in_json_values(json_string: str) -> str:
    result = ""
    i_am_inside_quotes = False
    for v in json_string:
        if v == '"':
            i_am_inside_quotes = not i_am_inside_quotes
        if v == "\n" and i_am_inside_quotes:
            result += "\\n"
        else:
            result += v
    return result


def close_all_curly_brackets(json_string: str) -> str:
    result = json_string
    num_brackets_to_close = json_string.count("{") - json_string.count("}")
    result += "}" * num_brackets_to_close
    return result


class ThoughtsJSONParser(BaseOutputParser):
    thoughts: List[Thought]
    stop_sequences: List[str] = ["}\n```", "}```"]
    format_prompt: str = FORMAT_PROMPT

    @validator("thoughts")
    def validate_thoughts(cls, v):
        assert len(v) > 0, "You must have at least one thought"
        return v

    def parse(self, text: str) -> Dict[str, Any]:
        cleaned_output = text.strip()
        if "```json" in cleaned_output:
            cleaned_output = cleaned_output[cleaned_output.find("```json") + len("```json"):]
        cleaned_output = cleaned_output.strip()
        cleaned_output = escape_new_lines_in_json_values(cleaned_output)
        cleaned_output = close_all_curly_brackets(cleaned_output)
        response = json.loads(cleaned_output)
        return response

    def format_thoughts(self) -> str:
        return "\n\t".join([
            f'"{thought.name}": {thought.type} [{thought.description}]' for thought in self.thoughts
        ])

    def get_format_instructions(self) -> str:
        format_instructions = PromptTemplate.from_template(self.format_prompt, template_format="jinja2").format_prompt(
            thoughts=self.format_thoughts()).to_string()
        return format_instructions
