import unittest

from yid_langchain_extensions.output_parser.action_parser import ActionParser
from yid_langchain_extensions.output_parser.class_parser import Class, get_classes_description, get_classes_summary, \
    ClassParser
from yid_langchain_extensions.output_parser.thoughts_json_parser import ThoughtsJSONParser, Thought


class TestThoughtsJSONParser(unittest.TestCase):
    def test_parse(self):
        parser = ThoughtsJSONParser(
            extra_thoughts=[
                Thought(name="thought1", description="Thought 1"),
                Thought(name="thought2", type="int", description="Thought 2")
            ]
        )
        string_to_parse = """```json
{
    "thought1": "thought 1",
    "thought2": 2
"""
        parsed = parser.parse(string_to_parse)
        self.assertEqual(parsed, {
            "thought1": "thought 1",
            "thought2": 2
        })

    def test_format_thoughts(self):
        thoughts = [
            Thought(name="thought1", description="Thought 1"),
            Thought(name="thought2", type="int", description="Thought 2")
        ]
        thoughts_string = ThoughtsJSONParser(
            thoughts=thoughts
        ).format_thoughts()
        self.assertEqual(thoughts_string, '"thought1": string [Thought 1]\n\t"thought2": int [Thought 2]')

    def test_format_instructions(self):
        thoughts = [
            Thought(name="thought1", description="Thought 1"),
            Thought(name="thought2", type="int", description="Thought 2")
        ]
        format_instructions = ThoughtsJSONParser(
            thoughts=thoughts
        ).get_format_instructions()
        self.assertTrue('"thought1": string [Thought 1]\n\t"thought2": int [Thought 2]' in format_instructions)


class TestClassParser(unittest.TestCase):
    def setUp(self) -> None:
        self.classes = [
            Class(name="class1", description="description1"),
            Class(name="class2", description="description2"),
        ]

    def test_get_classes_description(self):
        self.assertEqual(get_classes_description(self.classes),
                         "> 0: class1. description1;\n> 1: class2. description2;\n")

    def test_get_classes_summary(self):
        self.assertEqual(get_classes_summary(self.classes), "0 (class1); 1 (class2)")

    def test_class_parser(self):
        class_parser = ClassParser(extra_thoughts=[
            Thought(name="thought1", description="Thought 1")
        ])
        string_to_parse = """```json
{
    "thought1": "thought 1",
    "class_index": 0
"""
        self.assertEqual(class_parser.parse(string_to_parse), 0)

    def test_extra_thoughts(self):
        extra_thoughts = [
            Thought(name="thought1", description="Thought 1"),
            Thought(name="thought2", type="int", description="Thought 2")
        ]
        class_parser = ClassParser.from_extra_thoughts(extra_thoughts=extra_thoughts)
        self.assertEqual(class_parser.thoughts[:2], extra_thoughts)
        self.assertEqual(class_parser.thoughts[2].name, "class_index")


class TestActionParser(unittest.TestCase):
    def test_class_parser(self):
        class_parser = ActionParser(extra_thoughts=[
            Thought(name="thought1", description="Thought 1")
        ])
        string_to_parse = """```json
{
    "thought1": "thought 1",
    "action": "action 1",
    "action_input": "action input"
"""
        action = class_parser.parse(string_to_parse)
        self.assertEqual(action.tool, "action 1")
        self.assertEqual(action.tool_input, "action input")

    def test_extra_thoughts(self):
        extra_thoughts = [
            Thought(name="thought1", description="Thought 1"),
            Thought(name="thought2", type="int", description="Thought 2")
        ]
        class_parser = ActionParser.from_extra_thoughts(extra_thoughts=extra_thoughts)
        self.assertEqual(class_parser.thoughts[:2], extra_thoughts)
        self.assertEqual(class_parser.thoughts[2].name, "action")
        self.assertEqual(class_parser.thoughts[3].name, "action_input")
