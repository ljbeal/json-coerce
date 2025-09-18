from pydantic import ValidationError
import pytest
from json_coerce.structures import GeneratedFunction


invalid = [
    """def generate_random_integers(n):
    return [randint(1, 100) for _ in range(n)]

from random import randint"""
]


@pytest.mark.parametrize("case", invalid)
def test_invalid_cases(case: str) -> None:
    print("Validating case:")
    print(case)
    structure = GeneratedFunction

    with pytest.raises(ValidationError):
        structure.model_validate({"name": "generate_random_integers", "source": case})
