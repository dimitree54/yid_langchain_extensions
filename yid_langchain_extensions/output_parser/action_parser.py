from typing import List, Dict, Any, Union

from langchain.agents import AgentOutputParser
from langchain.schema import AgentAction

from yid_langchain_extensions.output_parser.thoughts_json_parser import Thought, ThoughtsJSONParser


class ActionWithThoughts(AgentAction):
    tool_input: Union[str, dict, list]
    all_thoughts: Dict[str, Any]


class ActionParser(ThoughtsJSONParser, AgentOutputParser):
    @classmethod
    def get_action_thoughts(cls) -> List[Thought]:
        return [
            Thought(name="action", description="The action to take. Must be one of [{{tool_names}}]"),
            Thought(name="action_input", description="The input to the action")
        ]

    @classmethod
    def from_extra_thoughts(cls, pre_thoughts: List[Thought], after_thoughts: List[Thought]):
        thoughts = pre_thoughts + cls.get_action_thoughts() + after_thoughts
        return cls(thoughts=thoughts)

    def parse(self, text: str) -> ActionWithThoughts:
        response = super().parse(text)
        fixed_text = self.fix_json_md_snippet(text)
        return ActionWithThoughts(
            tool=response["action"], tool_input=response["action_input"], log=fixed_text, all_thoughts=response)
