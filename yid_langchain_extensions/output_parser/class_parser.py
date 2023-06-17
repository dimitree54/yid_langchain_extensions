from typing import List

from pydantic import BaseModel

from yid_langchain_extensions.output_parser.thoughts_json_parser import Thought, ThoughtsJSONParser


class Class(BaseModel):
    name: str
    description: str


def get_classes_description(classes: List[Class]) -> str:
    result = ""
    for class_index, _class in enumerate(classes):
        result += f"> {class_index}: {_class.name}. {_class.description};\n"
    return result


def get_classes_summary(classes: List[Class]) -> str:
    return "; ".join([f"{class_index} ({_class.name})" for class_index, _class in enumerate(classes)])


class ClassParser(ThoughtsJSONParser):
    extra_thoughts: List[Thought] = []
    action_thoughts: List[Thought] = [
        Thought(name="class_index", type="int", description="The class chosen. Must be one of [{classes_summary}]"),
    ]

    @property
    def thoughts(self) -> List[Thought]:
        return self.action_thoughts + self.extra_thoughts

    def parse(self, text: str) -> int:
        response = super().parse(text)
        return int(response["class_index"])
