import json
from typing import Dict, List


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
    result += "\n}" * num_brackets_to_close
    return result


def strip_json_from_md_snippet(json_md_snippet: str) -> str:
    cleaned_text = json_md_snippet[len("```json"): -len("```")]
    return cleaned_text


def get_dict_without_extra_fields(dict_to_strip: Dict, keys_to_keep: List[str]) -> Dict:
    stripped_dict = {}
    for key, value in dict_to_strip.items():
        if key in keys_to_keep:
            stripped_dict[key] = value
    return stripped_dict


def format_dict_to_json_md(dict_to_format: Dict) -> str:
    return "```json\n" + json.dumps(dict_to_format, indent=4) + "\n```"
