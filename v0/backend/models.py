from typing import Dict, List
from pydantic import BaseModel


class ActionData(BaseModel):
    input_json_schema: str
    output_json_schema: str
    code: str
    test: str
    chat_history: List[Dict[str, str]]  # List of chat messages


class ActionDataWeaviate(ActionData):
    text_to_embed: str


class ActionDataWeaviateScored(ActionDataWeaviate):
    score: float


class RetrievalRequest(BaseModel):
    chat_history: List[Dict[str, str]]
    threshold: float = 0.9
    top_k: int = 5
