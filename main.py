from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion

from json_coerce.structures import GeneratedFunction
from json_coerce.wrapper import StructuredWrapper


client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # required, but unused
)

# "Mark is a 34 year old software engineer living in San Francisco."


def generate_response(prompt: str, model: str, structure: str) -> ChatCompletion:
    modified_prompt = f"""Extract details or perform tasks according to the following text:

"{prompt}"

Your response should contain ONLY a valid JSON object with the following fields.
Do not respond with any other content, only the JSON object with the following fields:
{structure}
"""

    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": modified_prompt}],
    )

    return completion


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
