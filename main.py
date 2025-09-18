from openai import OpenAI

from json_coerce.structures import GeneratedFunction
from json_coerce.wrapper import StructuredWrapper


client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # required, but unused
)


if __name__ == "__main__":
    models = [
        "gpt-oss",
        "deepseek-coder-v2:16b",
        "codestral",  # tool bug
        "mixtral",
        "mistral",
    ]

    functionchat = StructuredWrapper(client, GeneratedFunction)

    prompt = "Write a Python function that returns n random integers between 1 and 100."

    print("### Testing generation with various models ###")
    print(f"models: {models}")
    print(f"User Prompt:\n{prompt}\n")
    print(f"Structure:\n{functionchat.structure}\n")

    for model in models:
        print(f"\n### Generating with model: {model}")
        result = functionchat.chat(prompt, model)

        print("Result:")
        print(f"Name: {result.get('name')}")
        print(f"Source:\n{result.get('source')}")
