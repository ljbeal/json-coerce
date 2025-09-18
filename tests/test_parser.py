import json
import pytest

from json_coerce.json_parser import clean_output


cases = [
    '{"name": "John", "age": 30}',  # Valid JSON,
    '{\n\t"name": "John",\n\t"age": 30\n}',  # JSON with newlines and tabs
    "```json\n{\"name\": \"John\", \"age\": 30}\n```",  # JSON in markdown
]


@pytest.mark.parametrize("case", cases)
def test_cases(case: str) -> None:
    print("#-- input --#")
    print(case)
    cleaned = clean_output(case)
    print("#-- cleaned --#")
    print(cleaned)
    assert json.loads(cleaned) == {"name": "John", "age": 30}
