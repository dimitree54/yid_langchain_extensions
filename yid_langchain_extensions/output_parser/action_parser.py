from typing import List

from langchain.schema import AgentAction

from yid_langchain_extensions.output_parser.thoughts_json_parser import Thought, ThoughtsJSONParser


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
