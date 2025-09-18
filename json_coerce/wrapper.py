"""
------------------------------------------
Copyright: CEA Grenoble
Auteur: Louis BEAL
Entité: IRIG
Année: 2025
Description: Agent IA d'Intégration Continue
------------------------------------------
"""

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

        self.history = []

    def _chat(self, model: str, prompt: str | None = None) -> str:
        """
        Act on the current message history, appending the response to the history.
        """
        if prompt is not None and (
            len(self.history) == 0 or self.history[-1]["role"] != "user"
        ):
            self.history.append({"role": "user", "content": prompt})

        if len(self.history) > 0 and self.history[-1]["role"] != "user":
            raise ValueError("Last message in history must be from user")

        completion = self.client.chat.completions.create(
            model=model,
            messages=self.history,
        )

        if len(completion.choices) == 0:
            return ""
        if completion.choices[0].message is None:
            return ""
        if completion.choices[0].message.content is None:
            return ""

        self.history.append(
            {"role": "assistant", "content": completion.choices[0].message.content}
        )

        return completion.choices[0].message.content

    def _get_json(
        self, text: str, current_retries: int, max_retries: int, retry_model: str
    ) -> dict[str, str]:
        """
        Attempt to extract JSON from the text, retrying if necessary.
        """

        try:
            return json.loads(clean_output(text))
        except json.JSONDecodeError:
            if current_retries >= max_retries:
                raise ValueError(
                    f"Max retries reached, unable to extract valid JSON. Last attempt:\n{text}"
                )
            print(
                f"Failed to parse JSON, asking {retry_model} to retry... (try {current_retries + 1}/{max_retries})"
            )
            retry = self._chat(
                model=retry_model, prompt=JSON_RETRY_PROMPT.format(input=text)
            )
            return self._get_json(retry, current_retries + 1, max_retries, retry_model)

    def _validate_structure(
        self,
        data: dict[str, str],
        current_retries: int,
        max_retries: int,
        retry_model: str,
    ) -> dict[str, str]:
        try:
            self.structure_model.model_validate(data)
            return data

        except ValidationError as e:
            if current_retries >= max_retries:
                raise ValueError(
                    f"Max retries reached, unable to produce valid structure. Last attempt:\n{json.dumps(data, indent=2)}"
                )
            print(
                f"Validation failed, asking {retry_model} to fix the issue... (try {current_retries + 1}/{max_retries})"
            )
            retry = self._chat(
                model=retry_model,
                prompt=VALIDATION_RETRY_PROMPT.format(
                    output=json.dumps(data, indent=2),
                    error=str(e),
                    structure=self.structure,
                ),
            )
            parsed = json.loads(clean_output(retry))
            return self._validate_structure(
                parsed, current_retries + 1, max_retries, retry_model
            )

    def chat(
        self, prompt: str | list[dict[str, str]], model: str, max_retries: int = 3
    ) -> dict[str, str]:
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]

        current_retries = 0

        # get the content of the last user prompt
        user_prompt_content = prompt.pop(-1)["content"]
        # inject our structure into the prompt
        modified_prompt = f"""Extract details or perform tasks according to the following text:

"{user_prompt_content}"

Your response should contain ONLY a valid JSON object with the following fields.
Do not respond with any other content, only the JSON object with the following fields:
{self.structure}"""

        # get a response
        result = self._chat(model=model, prompt=modified_prompt)
        if result == "":
            return {}

        # now parse the json
        parsed = self._get_json(result, current_retries, max_retries, retry_model=model)
        # validate the model structure
        parsed = self._validate_structure(
            parsed, current_retries, max_retries, retry_model=model
        )

        return parsed

    @property
    def last_response(self) -> str:
        """
        Get the content of the last assistant response
        """
        for message in self.history[::-1]:
            if message["role"] == "assistant":
                return message["content"]
        return ""
