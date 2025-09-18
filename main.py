import json
from openai import OpenAI
from pydantic import BaseModel, field_validator

from json_coerce.structures import GeneratedFunction
from json_coerce.wrapper import StructuredWrapper


client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # required, but unused
)


class Person(BaseModel):
    name: str
    age: int
    city: str
    employed: bool

    @field_validator("age")
    def age_must_be_positive(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Age must be a positive integer")
        return v


if __name__ == "__main__":
    models = [
        "gpt-oss",
        "deepseek-coder-v2:16b",
        "codestral",  # tool bug
        "mixtral",
        "mistral",
    ]

    functionchat = StructuredWrapper(client, GeneratedFunction)

    prompt = "Write a Python function that returns n random characters from the hexadecimal set"

    # prompt = "Jason is a 30 year old software engineer living in San Francisco."

    print("### Testing generation with various models ###")
    print(f"models: {models}")
    print(f"User Prompt:\n{prompt}\n")
    print(f"Structure:\n{functionchat.structure}\n")

    for model in models:
        try:
            print(f"\n### Generating with model: {model}")
            result = functionchat.chat(prompt, model)

            print("Result:")
            print(json.dumps(result, indent=2))
            # print(f"Name: {result.get('name')}")
            print(f"Source:\n{result.get('source')}")
        finally:
            with open(f"chat_log_{model}.txt", "w") as f:
                for i, message in enumerate(functionchat.history):
                    f.write(f"### Message {i + 1} ###\n")
                    f.write(f"{message['role'].upper()}:\n{message['content']}\n\n")
