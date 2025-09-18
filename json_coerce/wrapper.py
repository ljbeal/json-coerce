import json
from openai import OpenAI
from pydantic import BaseModel

from json_coerce.json_parser import clean_output
from json_coerce.model_convert import convert_model_to_struct


RETRY_PROMPT = """Given the following input, extract relevant content and ensure it is formatted as valid JSON.

Strip away any external formatting, markdown or commentary and return only the inner content.

Return only the output of this operation, do not add your own commentry, explanation or formatting.
{input}
"""


class StructuredWrapper:

    def __init__(self, client: OpenAI, structure: BaseModel.__class__) -> None:
        self.client = client
        self.structure_model = structure
        self.structure = convert_model_to_struct(structure)
        
    def _chat(self, prompt: str, model: str) -> str:
        print(f"chat prompt:\n'{prompt}'")
        completion = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        if len(completion.choices) == 0:
            return ""
        if completion.choices[0].message is None:
            return ""
        if completion.choices[0].message.content is None:
            return ""

        return completion.choices[0].message.content

    def chat(self, prompt: str, model: str) -> dict[str, str]:
        modified_prompt = f"""Extract details or perform tasks according to the following text:

"{prompt}"

Your response should contain ONLY a valid JSON object with the following fields.
Do not respond with any other content, only the JSON object with the following fields:
{self.structure}"""

        result = self._chat(modified_prompt, model)
        if result is None:
            return {}

        # TODO: now parse the json
        try:
            parsed = json.loads(clean_output(result))
        except ValueError:
            print("Initial parse failed, retrying with stricter prompt...")
            parsed = json.loads(clean_output(self._chat(RETRY_PROMPT.format(input=result), model=model)))
            
        self.structure_model.model_validate(parsed)
        return parsed
