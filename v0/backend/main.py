import json
from fastapi import FastAPI
from weaviate_service import WeaviateClient
from dotenv import load_dotenv
from typing import List
from models import ActionData, RetrievalRequest


# Load environment variables from .env file
load_dotenv()

app = FastAPI()
client = WeaviateClient()


@app.post("/submit_action")
async def submit_action(submission: ActionData) -> bool:
    """
    1. Get embeddings for chat history
    2. Store action in Weaviate with chat history embedding
    3. Return success/failure
    """
    # TODO: Implement embedding generation using OpenAI
    # TODO: Store in Weaviate
    client.add_action_data(submission)
    return True


@app.post("/retrieve_actions")
async def retrieve_actions(request: RetrievalRequest) -> List[ActionData]:
    """
    1. Get embeddings for input chat history
    2. Query Weaviate for similar actions
    3. Return top k results
    """
    # TODO: Implement embedding generation and retrieval
    action_data_tuples = client.retrieve_action_data(
        json.dumps(request.chat_history), request.top_k
    )
    print("action_data_tuples:\n", action_data_tuples)
    action_data_objects = [
        action_data_tuple[0]
        for action_data_tuple in action_data_tuples
        if action_data_tuple[1] > request.threshold
    ]
    return action_data_objects


# delete collection
@app.delete("/delete_collection")
async def delete_collection():
    client.delete_collection("actions")
    return True