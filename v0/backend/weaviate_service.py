# src/services/weaviate_client.py

import json
from typing import List, cast

from pydantic import BaseModel
import weaviate
from weaviate.classes.config import Configure
from weaviate.classes.query import MetadataQuery
from models import ActionData, ActionDataWeaviate, ActionDataWeaviateScored
import os


def action_data_to_weaviate_item(action_data: ActionData) -> ActionDataWeaviate:
    return ActionDataWeaviate.model_validate(
        {
            **action_data.model_dump(),
            "text_to_embed": json.dumps(action_data.chat_history),
        }
    )


def weaviate_item_to_action_data(weaviate_item: ActionDataWeaviate) -> ActionData:
    return ActionData.model_validate(weaviate_item)


def graded_weaviate_item_to_action_data(
    weaviate_item: ActionDataWeaviateScored,
) -> ActionData:
    return ActionData.model_validate(weaviate_item)


class WeaviateClient:
    def __init__(self):
        print("VOYAGEAI_API_KEY", os.getenv("VOYAGEAI_API_KEY"))
        headers = {
            "X-VoyageAI-Api-Key": os.getenv("VOYAGEAI_API_KEY", ""),
        }
        if not headers["X-VoyageAI-Api-Key"]:
            raise ValueError("Please provide a VoyageAI API key")
        self.client = weaviate.connect_to_local(
            host=os.getenv("WEAVIATE_HOST", "localhost"),
            port=int(os.getenv("WEAVIATE_PORT", "8080")),
            grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
            headers=headers,
        )
        meta_info = self.client.get_meta()
        print(meta_info)

    def ensure_collection(self, collection_name: str) -> None:
        if self.client.collections.exists(collection_name):
            print(f"Collection '{collection_name}' already exists")
        else:
            self.client.collections.create(
                collection_name,
                vectorizer_config=[
                    Configure.NamedVectors.text2vec_voyageai(
                        name="title_vector",
                        source_properties=["text_to_embed"],
                        model="voyage-2",
                    )
                ],
            )
            print(f"Collection '{collection_name}' created successfully")

    def add_action_data(self, action_data: ActionData) -> None:
        collection_name = "actions"
        self.ensure_collection(collection_name)
        collection = self.client.collections.get(collection_name)
        weaviate_item = action_data_to_weaviate_item(action_data)
        collection.data.insert(weaviate_item.model_dump())
        print(f"Added:{weaviate_item.model_dump_json(indent=4)[:50] + '...'} to collection '{collection_name}'")

    def retrieve_action_data(
        self, query: str, top_k: int = 10
    ) -> List[tuple[ActionData, float]]:
        collection_name = "actions"
        self.ensure_collection(collection_name)
        collection = self.client.collections.get(collection_name)
        try:    
            response = collection.query.hybrid(
                query=query,
                limit=top_k,
                include_vector=False,
                return_metadata=MetadataQuery(score=True),
            )
        except Exception as e:
            print(f"Error retrieving actions: {e}")
            return []
        action_data_tuples = []
        for obj in response.objects:
            if obj is None:
                continue
            else:
                action_data_tuples.append(
                    (
                        ActionData(**obj.properties),
                        obj.metadata.score,
                    )
                )
        return action_data_tuples

    # def add_target_clients(
    #     self, session_id: str, chunk_target_client: List[ChunkTargetClient]
    # ) -> None:
    #     collection_name = f"{session_id}_targets"
    #     self.ensure_collection(collection_name)
    #     collection = self.client.collections.get(collection_name)
    #     with collection.batch.dynamic() as batch:
    #         for profile in chunk_target_client:
    #             batch.add_object(profile.model_dump())
    #     if len(collection.batch.failed_objects) > 0:
    #         print(f"Failed to import {len(collection.batch.failed_objects)} objects")
    #         for failed in collection.batch.failed_objects:
    #             print(f"Failed to import object with error: {failed.message}")
    #     print(
    #         f"Added {len(chunk_target_client) - len(collection.batch.failed_objects)} objects to collection '{collection_name}'"
    #     )

    # def retrieve_target_clients(
    #     self, session_id: str, query: str
    # ) -> List[GradedTargetClient]:
    #     collection_name = f"{session_id}_targets"
    #     if not self.client.collections.exists(collection_name):
    #         raise ValueError(f"Collection '{collection_name}' does not exist")
    #     collection = self.client.collections.get(collection_name)
    #     response = collection.query.hybrid(
    #         query=query,
    #         limit=100,
    #         include_vector=False,
    #         return_metadata=MetadataQuery(score=True),
    #     )
    #     graded_target_client_objects = []
    #     for obj in response.objects:
    #         if obj is None:
    #             continue
    #         else:
    #             graded_target_client = {
    #                 "chunk": obj.properties,
    #                 "grade": obj.metadata.score,
    #             }
    #             graded_target_client_objects.append(
    #                 GradedTargetClient(**graded_target_client)
    #             )

    #     return graded_target_client_objects

    def delete_collection(self, collection_name: str) -> None:
        if not self.client.collections.exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")
        self.client.collections.delete(collection_name)
        print(f"Collection '{collection_name}' deleted successfully")
