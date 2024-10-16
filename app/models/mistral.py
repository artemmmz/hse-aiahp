import time
from enum import Enum
from typing import Optional

import requests

from app.models.base import BaseModel


# Free mistral models
class MistralModels(str, Enum):
    NEMO = 'open-mistral-nemo'
    CODESTRAL_MAMDA = 'open-codestral-mamda'
    MISTRAL_7B = 'open-mistral-7b'
    MISTRAL_8X7B = 'open-mistral-8x7b'
    MISTRAL_8x22B = 'open-mistral-8x22b'


class Mistral(BaseModel):
    def __init__(
            self,
            token: str,
            model: MistralModels = MistralModels.NEMO,
            system_prompt: Optional[str] = None,
    ):
        """

        """
        super().__init__(system_prompt)
        self.clean_history()
        self.token = token
        self.model = model.value

    def add_user_message(self, user_message: str):
        self.messages.append(
            {
                'role': 'user',
                'content': user_message,
            }
        )

    def add_assistant_message(self, assistant_message: str):
        self.messages.append(
            {
                'role': 'assistant',
                'content': assistant_message,
            }
        )

    def clean_history(self):
        self.messages = []
        if self.system_prompt:
            self.messages.append(
                {
                    'role': 'system',
                    'content': self.system_prompt,
                }
            )

    def ask(self, user_message: str, clean_history: bool = True) -> str | None:
        if clean_history:
            self.clean_history()

        url = 'https://api.mistral.ai/v1/chat/completions'
        header = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        data = {
            'model': self.model,
            'messages': self.messages,
        }

        response = requests.post(url, headers=header, json=data, verify=False)

        if response.status_code == 429:
            time.sleep(3)
            return self.ask(user_message, clean_history)
        if response.status_code != 200:
            return None

        data = response.json()
        answer = data['choices'][0]['message']['content']

        self.add_user_message(user_message)
        self.add_assistant_message(answer)

        return answer

