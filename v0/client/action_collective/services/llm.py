from typing import Any, Dict, List
import openai
from ..models.requests import ActionCollectiveRequest, ActionDataGenerator
from ..models.actions import ActionData

class LLMService:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
    
    async def get_action_thought(self, chat_history: List[Dict[str, str]]) -> ActionCollectiveRequest:
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=chat_history,
            response_format=ActionCollectiveRequest
        )
        return completion.choices[0].message.parsed
    
    async def generate_action(self, chat_history: List[Dict[str, str]]) -> ActionDataGenerator:
        response = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=chat_history,
            response_format=ActionDataGenerator
        )
        return response.choices[0].message.parsed