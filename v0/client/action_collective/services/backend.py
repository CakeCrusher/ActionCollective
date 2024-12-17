import requests
from typing import List, Optional
from ..models.actions import ActionData

class BackendService:
    def __init__(self, backend_url: str):
        self.backend_url = backend_url
    
    async def submit_action(self, action: ActionData) -> bool:
        response = requests.post(
            f"{self.backend_url}/submit_action",
            json=action.model_dump()
        )
        return response.json()
    
    async def retrieve_actions(
        self, 
        chat_history: List[dict], 
        top_k: int = 5, 
        threshold: float = 0.7
    ) -> List[ActionData]:
        response = requests.post(
            f"{self.backend_url}/retrieve_actions",
            json={"chat_history": chat_history, "top_k": top_k, "threshold": threshold}
        )
        return [ActionData.model_validate(action) for action in response.json()]