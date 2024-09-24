from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, conlist

from yid_langchain_extensions.output_parser.fixing_json_parser import FixingJSONParser

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


class ThoughtsJSONParser(FixingJSONParser):
    thoughts: conlist(Thought, min_length=1)
    format_prompt: str = FORMAT_PROMPT

    def format_thoughts(self) -> str:
        return "\n\t".join([
            f'"{thought.name}": {thought.type} [{thought.description}]' for thought in self.thoughts
        ])

    def get_format_instructions(self, **kwargs) -> str:
        """
        :param kwargs: If your thoughts or format_prompt have some extra placeholders, you can fill them by kwargs
        """
        format_instructions = PromptTemplate.from_template(self.format_prompt, template_format="jinja2").partial(
            thoughts=self.format_thoughts()).format_prompt(**kwargs).to_string()
        return format_instructions
