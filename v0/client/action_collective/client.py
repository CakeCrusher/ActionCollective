from typing import Optional, List, Dict
from .services.llm import LLMService
from .services.backend import BackendService
from .models.actions import ActionData, ActionExecutionPayload
import os
from openai._exceptions import LengthFinishReasonError
import json


class ActionClient:
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        backend_url: Optional[str] = None,
        verbose: bool = False,
    ):
        self.llm = LLMService(openai_api_key or os.getenv("OPENAI_API_KEY"))
        self.backend = BackendService(backend_url or os.getenv("BACKEND_URL"))
        self.chat_history: List[Dict[str, str]] = []
        self.internal_chat_history: List[Dict[str, str]] = []
        self.verbose = verbose

    async def validate_schema(self, schema: dict) -> None:
        """Validate the schema using OpenAI's parse endpoint"""
        try:
            if not schema.get("description"):
                raise Exception("Description is required for all properties")

            self.llm.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "a"}],
                max_completion_tokens=1,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "action_items",
                        "description": "The action items to be completed",
                        "strict": True,
                        "schema": schema,
                    },
                },
            )
        except LengthFinishReasonError as e:
            pass
        except Exception as e:
            raise Exception(f"Failed to validate schema: {e}")

    async def execute(self, prompt: str, max_retries: int = 3) -> str:
        """
        Main entry point for executing actions based on a prompt
        """
        # Add user message to chat history
        self.chat_history.append({"role": "user", "content": prompt})

        # Get action thought
        action_thought = await self.llm.get_action_thought(self.chat_history)

        if not action_thought.is_action_needed:
            return action_thought.thought

        # Record the thought process
        self.chat_history.append(
            {
                "role": "assistant",
                "content": f"THOUGHT:\n{action_thought.thought}\nTOOL DESCRIPTION:\n{action_thought.tool_description}",
            }
        )

        # Try to retrieve existing action or create new one
        actions = await self.backend.retrieve_actions(self.chat_history)

        if self.verbose:
            print("\n\nretrieve_actions:\n", actions, "\n\n")

        if actions:
            action = actions[0]
        else:
            # Generate new action with retries
            retries = 0
            while retries < max_retries:
                try:
                    action_generator = await self.llm.generate_action(
                        self.chat_history + self.internal_chat_history
                    )

                    # MITIGATE COMMON ERROR: add additionalProperties to the input_json_schema
                    action_generator.input_json_schema = json.dumps(
                        {
                            **json.loads(action_generator.input_json_schema),
                            "additionalProperties": False,
                        }
                    )
                    action_generator.output_json_schema = json.dumps(
                        {
                            **json.loads(action_generator.output_json_schema),
                            "additionalProperties": False,
                        }
                    )

                    # Clean up code blocks
                    action_generator.code = (
                        action_generator.code.replace("```python", "")
                        .replace("```", "")
                        .strip()
                    )
                    action_generator.test = (
                        action_generator.test.replace("```python", "")
                        .replace("```", "")
                        .strip()
                    )

                    if self.verbose:
                        print(
                            "\n\ngenerate_action POST FIX:\n",
                            action_generator.model_dump_json(indent=4),
                        )

                    # Validate schema and test code
                    complete_test = action_generator.code + "\n" + action_generator.test
                    loaded_schema = json.loads(action_generator.input_json_schema)
                    await self.validate_schema(loaded_schema)

                    print("\n\nexecuting:\n\n", complete_test)
                    exec(complete_test, {})
                    print("\n\nPASSED")

                    action = ActionData(
                        **action_generator.model_dump(), chat_history=self.chat_history
                    )
                    await self.backend.submit_action(action)
                    break

                except Exception as e:
                    print(f"\n\nRETRY {retries} FAILED", e)
                    self.internal_chat_history.append(
                        {
                            "role": "assistant",
                            "content": f"""Action data content
```
{action_generator.model_dump_json()}
```
RETRY {retries} FAILED: {e}
Make sure to maintain a simple JSON Schema as in the example.""",
                        }
                    )
                    retries += 1

            if retries >= max_retries:
                raise Exception("Failed to create valid action after maximum retries")

        # Execute action and return results
        # TODO: Implement action execution
        return f"Action executed: {action.code[:100]}..."
