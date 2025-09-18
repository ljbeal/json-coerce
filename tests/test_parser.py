import json
import pytest

from json_coerce.json_parser import clean_output


cases = [
    '{"name": "John", "age": 30}',  # Valid JSON,
    '{\n\t"name": "John",\n\t"age": 30\n}',  # JSON with newlines and tabs
    '{\\n"name": "John",\\n"age": 30\\n}',  # JSON with newlines escaped
    "```json\n{\"name\": \"John\", \"age\": 30}\n```",  # JSON in markdown
    """{
  "name": 'John', "age": 30
}

The Python function"""  # '' quotes
]


@pytest.mark.parametrize("case", cases)
def test_cases(case: str) -> None:
    print("#-- input --#")
    print(case)
    cleaned = clean_output(case)
    print("#-- cleaned --#")
    print(cleaned)
    assert json.loads(cleaned) == {"name": "John", "age": 30}
