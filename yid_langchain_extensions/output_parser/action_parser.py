from typing import List

from langchain.agents import AgentOutputParser
from langchain.schema import AgentAction

from yid_langchain_extensions.output_parser.thoughts_json_parser import Thought, ThoughtsJSONParser


class ActionParser(ThoughtsJSONParser, AgentOutputParser):
    @classmethod
    def get_action_thoughts(cls) -> List[Thought]:
        return [
            Thought(name="action", description="The action to take. Must be one of [{{tool_names}}]"),
            Thought(name="action_input", description="The input to the action")
        ]

    @classmethod
    def from_extra_thoughts(cls, extra_thoughts: List[Thought]):
        thoughts = extra_thoughts + cls.get_action_thoughts()
        return cls(thoughts=thoughts)

    def parse(self, text: str) -> AgentAction:
        response = super().parse(text)
        fixed_text = self.fix_json_md_snippet(text)
        return AgentAction(response["action"], response["action_input"], fixed_text)
