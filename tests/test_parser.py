import unittest

from pydantic import ValidationError

from yid_langchain_extensions.output_parser.action_parser import ActionParser
from yid_langchain_extensions.output_parser.class_parser import Class, get_classes_description, get_classes_summary, \
    ClassParser
from yid_langchain_extensions.output_parser.thoughts_json_parser import ThoughtsJSONParser, Thought


class TestThoughtsJSONParser(unittest.TestCase):
    def test_parse(self):
        parser = ThoughtsJSONParser(
            thoughts=[
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

    def test_parse_without_json(self):
        parser = ThoughtsJSONParser(
            thoughts=[
                Thought(name="thought1", description="Thought 1")
            ]
        )
        string_to_parse = """{
    "thought1": "thought 1"
}
"""
        parsed = parser.parse(string_to_parse)
        self.assertEqual(parsed, {
            "thought1": "thought 1"
        })

    def test_unpaired_brackets(self):
        parser = ThoughtsJSONParser(
            thoughts=[
                Thought(name="thought1", description="Thought 1")
            ]
        )
        string_to_parse = """```json
{
    "thought1": "thought 1"
}
"""
        parsed = parser.parse(string_to_parse)
        self.assertEqual(parsed, {
            "thought1": "thought 1"
        })

    def test_parse_multi_line_value(self):
        parser = ThoughtsJSONParser(
            thoughts=[
                Thought(name="thought1", description="Thought 1")
            ]
        )
        string_to_parse = """{
        "thought1": "thought
1"
    }
    """
        parsed = parser.parse(string_to_parse)
        self.assertEqual(parsed, {
            "thought1": "thought\n1"
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

    def test_complex_format_instructions(self):
        thoughts = [
            Thought(name="tool", description="One of [{{tools}}]"),
        ]
        format_instructions = ThoughtsJSONParser(
            thoughts=thoughts
        ).get_format_instructions(tools="tool1, tool2")
        self.assertTrue(format_instructions, '"tool": string [One of [tool1, tool2]]' in format_instructions)

    def test_raise_without_thoughts(self):
        with self.assertRaises(ValidationError):
            ThoughtsJSONParser(thoughts=[])

    def test_code_block(self):
        parser = ThoughtsJSONParser(
            thoughts=[
                Thought(name="action", description="action"),
                Thought(name="action_input", description="action_input")
            ]
        )
        string_to_parse = '''```json
        {
            "action": "Final Answer",
                "action_input": "Example:\\n```python\\nimport cv2\\n```\\n"
                '''
        parsed = parser.parse(string_to_parse)
        self.assertEqual(parsed["action_input"], "Example:\n```python\nimport cv2\n```\n")


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
        class_parser = ClassParser.from_extra_thoughts(extra_thoughts=[
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
    def test_action_parser(self):
        action_parser = ActionParser.from_extra_thoughts(pre_thoughts=[
            Thought(name="thought1", description="Thought 1")
        ], after_thoughts=[
            Thought(name="thought2", description="Thought 2")
        ])
        string_to_parse = """```json
{
    "thought1": "thought 1",
    "action": "action 1",
    "action_input": "action input",
    "thought2": "thought 2"
"""
        action = action_parser.parse(string_to_parse)
        self.assertEqual(action.tool, "action 1")
        self.assertEqual(action.tool_input, "action input")
        self.assertEqual(action.log, string_to_parse + "}\n```")
        self.assertEqual(action.all_thoughts["thought1"], "thought 1")
        self.assertEqual(action.all_thoughts["thought2"], "thought 2")

    def test_extra_thoughts(self):
        pre_thoughts = [
            Thought(name="thought1", description="Thought 1"),
            Thought(name="thought2", type="int", description="Thought 2")
        ]
        after_thoughts = [
            Thought(name="thought3", description="Thought 5"),
            Thought(name="thought4", type="int", description="Thought 6")
        ]
        action_parser = ActionParser.from_extra_thoughts(pre_thoughts=pre_thoughts, after_thoughts=after_thoughts)
        self.assertEqual(action_parser.thoughts[:2], pre_thoughts)
        self.assertEqual(action_parser.thoughts[2].name, "action")
        self.assertEqual(action_parser.thoughts[3].name, "action_input")
        self.assertEqual(action_parser.thoughts[4:], after_thoughts)

    def test_list_action_input(self):
        action_parser = ActionParser.from_extra_thoughts(pre_thoughts=[], after_thoughts=[])
        string_to_parse = """```json
{
    "action": "action 1",
    "action_input": ["action input 1", "action input 2"]
"""
        action = action_parser.parse(string_to_parse)
        self.assertEqual(action.tool, "action 1")
        self.assertEqual(action.tool_input,  ["action input 1", "action input 2"])
        self.assertEqual(action.log, string_to_parse + "}\n```")

    def test_list_action_dict(self):
        action_parser = ActionParser.from_extra_thoughts(pre_thoughts=[], after_thoughts=[])
        string_to_parse = """```json
{
    "action": "action 1",
    "action_input": {"parameter_name": "parameter_value"}
"""
        action = action_parser.parse(string_to_parse)
        self.assertEqual(action.tool, "action 1")
        self.assertEqual(action.tool_input,  {"parameter_name": "parameter_value"})
        self.assertEqual(action.log, string_to_parse + "}\n```")
