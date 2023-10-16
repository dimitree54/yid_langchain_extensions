from unittest import TestCase

from yid_langchain_extensions.output_parser.thoughts_json_parser import ThoughtsJSONParser
from yid_langchain_extensions.output_parser.utils import strip_json_from_md_snippet


class TestThoughtsJSONParser(TestCase):
    def setUp(self) -> None:
        self.json_md_snippet = """```json
{
    "thought1": "thought 1",
    "thought2": 2
}
```"""

    def test_strip_json_from_md_snippet(self):
        stripped_text = """
{
    "thought1": "thought 1",
    "thought2": 2
}
"""
        cleaned_text = strip_json_from_md_snippet(self.json_md_snippet)
        self.assertEqual(cleaned_text, stripped_text)

    def test_fix_json_md_snippet1(self):
        text = """```json
{
    "thought1": "thought 1",
    "thought2": 2"""
        fixed = ThoughtsJSONParser.fix_json_md_snippet(text)
        self.assertEqual(fixed, self.json_md_snippet)

    def test_fix_json_md_snippet2(self):
        text = """```json
{
    "thought1": "thought 1",
    "thought2": 2
}"""
        fixed = ThoughtsJSONParser.fix_json_md_snippet(text)
        self.assertEqual(fixed, self.json_md_snippet)

    def test_fix_json_md_snippet3(self):
        fixed = ThoughtsJSONParser.fix_json_md_snippet(self.json_md_snippet)
        self.assertEqual(fixed, self.json_md_snippet)

    def test_fix_json_md_snippet4(self):
        text = """
{
    "thought1": "thought 1",
    "thought2": 2"""
        fixed = ThoughtsJSONParser.fix_json_md_snippet(text)
        self.assertEqual(fixed, self.json_md_snippet)

    def test_fix_json_md_snippet5(self):
        text = """
{
    "thought1": "thought 1",
    "thought2": 2
}"""
        fixed = ThoughtsJSONParser.fix_json_md_snippet(text)
        self.assertEqual(fixed, self.json_md_snippet)

    def test_fix_json_md_snippet_extra_quotes(self):
        text = """"
{
    "thought1": "thought 1",
    "thought2": 2
}"""
        fixed = ThoughtsJSONParser.fix_json_md_snippet(text)
        self.assertEqual(fixed, self.json_md_snippet)
