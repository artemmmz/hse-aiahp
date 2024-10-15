import json
import os
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
from uuid import uuid4

import requests

from app.models.base import BaseModel


class GigachatModels(str, Enum):
    LITE = 'GigaChat'
    PRO = 'GigaChat-Pro'


class GigachatScopes(str, Enum):
    PERSON = 'GIGACHAT_API_PERS'
    B2B = 'GIGACHAT_API_B2B'
    CORPORATION = 'GIGACHAT_API_CORP'


class Gigachat(BaseModel):
    def __init__(
            self,
            token: str,
            client_id: str,
            model: GigachatModels = GigachatModels.LITE,
            scope: GigachatScopes = GigachatScopes.PERSON,
            system_prompt: Optional[str] = None,
            timeout: Optional[int] = 5000,
    ):
        """
        Initialize the Gigachat model.
        :param token: Gigachat API token
        :param model: Gigachat model
        :param scope: Gigachat scope
        :param system_prompt: System prompt
        :param timeout: Timeout
        """
        super().__init__(system_prompt)
        self.__token = token
        self.__model = model
        self.__scope = scope
        self.__client_id = client_id
        self.timeout = timeout

        self.__access_token = None
        self.token_expires = 0

        self.__update_access_key()

    def __update_access_key(self) -> None:
        """
        Update access key. USE ONLY AFTER INITIALIZATION
        """
        url = 'https://ngw.devices.sberbank.ru:9443/api/v2/oauth'
        data = f'scope={self.__scope.value}'
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json',
            'Authorization': f'Basic {self.__token}',
            'RqUID': str(uuid4()),
        }
        cert = os.path.abspath(os.path.join(os.getcwd(), "full_chain.cer"))

        response = requests.post(url, headers=headers, data=data, verify=False)

        if response.status_code != 200:
            return

        result = response.json()

        self.token_expires: int = result['expires_at']
        self.__access_token: str = result['access_token']

    def __check_access_key(self) -> None:
        """
        Check and update access_token if needed
        """
        timedout_now = datetime.now() + timedelta(seconds=self.timeout)
        if timedout_now.timestamp() >= self.token_expires - 1:
            self.__update_access_key()

    def ask(self, user_message: str, clear_history: bool = True) -> Optional[str]:
        """
        Send a message to the assistant and return the assistant's response.
        :param user_message: User message
        :param clear_history: True if history should be cleared
        """
        if clear_history:
            self.messages = []
        if len(self.messages) == 0 and self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

        self.messages.append({"role": "user", "content": user_message})
        self.__check_access_key()

        url = 'https://gigachat.devices.sberbank.ru/api/v1/chat/completions'
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.__access_token}',
        }
        json_request = {
            'model': self.__model,
            'messages': self.messages,
            'stream': False,
            'update_interval': 0
        }

        response = requests.post(
            url=url,
            headers=headers,
            data=json.dumps(json_request),
            timeout=self.timeout
        )
        if response.status_code != 200:
            return None

        result = response.json()
        assistant_content = result['choices'][0]['message']['content']
        self.messages.append({"role": "assistant", "content": assistant_content})

        return assistant_content
