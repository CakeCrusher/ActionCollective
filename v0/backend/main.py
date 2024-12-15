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
    action_data_objects = [action_data_tuple[0] for action_data_tuple in action_data_tuples]
    return action_data_objects


#     return [
#         ActionData(
#             input_json_schema=json.dumps(
#                 {
#                     "type": "object",
#                     "properties": {
#                         "A": {
#                             "type": "array",
#                             "description": "First input matrix",
#                             "items": {"type": "array", "items": {"type": "number"}},
#                         },
#                         "B": {
#                             "type": "array",
#                             "description": "Second input matrix",
#                             "items": {"type": "array", "items": {"type": "number"}},
#                         },
#                     },
#                     "required": ["A", "B"],
#                     "additionalProperties": False,
#                 }
#             ),
#             output_json_schema=json.dumps(
#                 {
#                     "type": "object",
#                     "properties": {
#                         "result": {
#                             "type": "array",
#                             "description": "The resulting matrix from A x B multiplication",
#                             "items": {"type": "array", "items": {"type": "number"}},
#                         }
#                     },
#                 }
#             ),
#             code="""
# import numpy as np

# def action(A: list[list[int]], B: list[list[int]]) -> list[list[int]]:
#     return np.dot(A, B)""",
#             test="""
# A = [[1, 2, 3, 4, 5],
#         [6, 7, 7, 9, 10],
#         [11, 12, 13, 14, 15],
#         [16, 17, 7, 19, 20],
#         [21, 22, 23, 24, 25]]
# B = [[1, 2, 3, 4, 5],
#         [6, 7, 8, 9, 10],
#         [11, 12, 7, 14, 15],
#         [16, 17, 18, 19, 20],
#         [21, 22, 23, 24, 25]]
# assert action(A, B) == [[215, 230, 227, 260, 275], [479, 518, 515, 596, 635], [765, 830, 817, 960, 1025], [919, 998, 1035, 1156, 1235], [1315, 1430, 1407, 1660, 1775]]""",
#             chat_history=[],
#         )
#     ]
