import json
import pytest

from pydantic import BaseModel
from json_coerce.model_convert import convert_model_to_struct


class SimpleModel(BaseModel):
    name: str


class LargeModel(BaseModel):
    name: str
    age: int
    height: float
    is_student: bool
    hobbies: list[str]


translate = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
}


@pytest.mark.parametrize("model", [SimpleModel, LargeModel])
def test_model_conversion(model: BaseModel.__class__) -> None:
    structure = convert_model_to_struct(model)
    assert structure.startswith("{")
    assert structure.endswith("}")

    clean = "\n".join(line for line in structure.split("\n") if "//" not in line)

    as_dict = json.loads(clean)

    for name, field in model.model_json_schema().get("properties", {}).items():
        assert name in as_dict
        assert translate[as_dict[name]] == field.get("type")
