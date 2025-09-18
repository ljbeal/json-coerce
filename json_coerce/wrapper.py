import json
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from json_coerce.json_parser import clean_output
from json_coerce.model_convert import convert_model_to_struct


JSON_RETRY_PROMPT = """Given the following input, extract relevant content and ensure it is formatted as valid JSON.

Strip away any external formatting, markdown or commentary and return only the inner content.

Return only the output of this operation, do not add your own commentry, explanation or formatting.
{input}"""

VALIDATION_RETRY_PROMPT = """You provided the following JSON:
{output}

However it fails to validate for the following reason:
{error}

Please ensure it fits the structure exactly, and return a corrected JSON object that validates correctly.
The structure is:
{structure}"""



class StructuredWrapper:

    def __init__(self, client: OpenAI, structure: BaseModel.__class__) -> None:
        self.client = client
        self.structure_model = structure
        self.structure = convert_model_to_struct(structure)
        
    def _chat(self, prompt: str, model: str) -> str:

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

    def chat(self, prompt: str, model: str, max_retries: int = 3) -> dict[str, str]:
        current_retries = 0

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
            while current_retries < max_retries:
                current_retries += 1
                print(f"JSON parse failed, asking {model} to clean it up... (try {current_retries}/{max_retries})")
                retry = self._chat(JSON_RETRY_PROMPT.format(input=result), model=model)
                parsed = json.loads(clean_output(retry))
        
        try:
            self.structure_model.model_validate(parsed)
        except ValidationError as e:
            while current_retries < max_retries:
                current_retries += 1
                print(f"Validation failed, asking {model} to fix the issue... (try {current_retries}/{max_retries})")
                retry = self._chat(
                    VALIDATION_RETRY_PROMPT.format(
                        output=parsed,
                        error=str(e),
                        structure=self.structure,
                    ),
                    model=model
                )
                parsed = json.loads(clean_output(retry))

        return parsed
