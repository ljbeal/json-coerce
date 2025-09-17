import json
from openai import OpenAI
from pydantic import BaseModel

from json_coerce.model_convert import convert_model_to_struct


class StructuredWrapper:

    def __init__(self, client: OpenAI, structure: BaseModel.__class__) -> None:
        self.client = client
        self.structure_model = structure
        self.structure = convert_model_to_struct(structure)

    def parse_output(str, content: str) -> dict:
        """
        Parse the JSON content from the LLM response.

        Args:
            content (str): The content string from the LLM response.

        Returns:
            dict: The parsed JSON object.
        """
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}")

    def chat(self, prompt: str, model: str) -> dict[str, str]:
        modified_prompt = f"""Extract details or perform tasks according to the following text:

    "{prompt}"

    Your response should contain ONLY a valid JSON object with the following fields.
    Do not respond with any other content, only the JSON object with the following fields:
    {self.structure}
    """
        
        completion = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": modified_prompt}
            ],
        )

        result = completion.choices[0].message.content
        if result is None:
            return {}

        # TODO: now parse the json
        parsed = self.parse_output(result)
        self.structure_model.model_validate(parsed)
        return parsed
