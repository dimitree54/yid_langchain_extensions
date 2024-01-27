import json
from typing import List, Dict, Any

from yid_langchain_extensions.output_parser.stop_seq_output_parser import StopSeqOutputParser
from yid_langchain_extensions.output_parser.utils import escape_new_lines_in_json_values, close_all_curly_brackets, \
    strip_json_from_md_snippet


class FixingJSONParser(StopSeqOutputParser):
    @property
    def stop_sequences(self) -> List[str]:
        return ["}\n```", "}```"]

    @staticmethod
    def fix_json_md_snippet(text: str) -> str:
        fixed_json = text
        if fixed_json.startswith("'") or fixed_json.startswith('"'):
            fixed_json = fixed_json[1:]
        fixed_json = fixed_json.strip()
        fixed_json = escape_new_lines_in_json_values(fixed_json)
        fixed_json = close_all_curly_brackets(fixed_json)
        if "```json" in fixed_json:
            fixed_json = fixed_json[fixed_json.find("```json"):]
        else:
            fixed_json = "```json\n" + fixed_json
        if not fixed_json.endswith("```"):
            fixed_json += "\n```"
        return fixed_json

    @staticmethod
    def parse_json_md_snippet(json_md_snippet: str) -> Dict[str, Any]:
        cleaned_json = strip_json_from_md_snippet(json_md_snippet)
        response = json.loads(cleaned_json)
        return response

    def parse(self, text: str) -> Dict[str, Any]:
        fixed_json_md_snippet = self.fix_json_md_snippet(text)
        return self.parse_json_md_snippet(fixed_json_md_snippet)
