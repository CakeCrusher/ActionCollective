from pydantic import BaseModel
from typing import List, Dict, Any


class ActionData(BaseModel):
    input_json_schema: str
    output_json_schema: str
    code: str
    test: str
    chat_history: List[Dict[str, str]]


class ActionExecutionPayload(BaseModel):
    action_data: ActionData
    params: Dict[str, Any]
