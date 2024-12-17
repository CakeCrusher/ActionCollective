from typing import Optional, List, Dict, Any
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

    async def build_action_execution_payload(
        self, action_data: ActionData
    ) -> ActionExecutionPayload:
        """Build execution payload with parameters from chat history"""
        if self.verbose:
            print("\n\nChat History Pre Params:\n", self.chat_history)

        action_params = self.llm.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=self.chat_history[:-1],  # Exclude the last assistant message
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "action_items",
                    "description": "The action items to be completed",
                    "strict": True,
                    "schema": json.loads(action_data.input_json_schema),
                },
            },
        )

        if not action_params.choices[0].message.content:
            raise Exception("Failed to get action params")

        params = json.loads(action_params.choices[0].message.content)

        if self.verbose:
            print("\n\nparams:\n", params)

        return ActionExecutionPayload(action_data=action_data, params=params)

    async def execute_action(
        self, action_execution_payload: ActionExecutionPayload
    ) -> Any:
        """Execute the action with the provided payload"""
        action_data = action_execution_payload.action_data
        params = action_execution_payload.params

        # Create a namespace for execution
        namespace = {}

        # Execute the action code to define the function in our namespace
        exec(action_data.code, namespace)

        # Execute the action function with unpacked parameters
        result = namespace["action"](**params)

        self.internal_chat_history.append(
            {"role": "assistant", "content": f"RESULT FROM ACTION: {result}"}
        )

        return result

    async def summarize_execution(self) -> str:
        """Summarize the execution result"""
        self.internal_chat_history.append(
            {"role": "assistant", "content": f"Now I will summarize the result..."}
        )

        summary_response = self.llm.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=self.chat_history + self.internal_chat_history,
        )

        summary = summary_response.choices[0].message.content
        self.internal_chat_history.append({"role": "assistant", "content": summary})

        return summary

    async def retrieve_or_generate(
        self, prompt: str, max_retries: int = 3
    ) -> ActionData:
        """Retrieve an existing action or generate a new one"""
        # Add user message to chat history
        self.chat_history.append({"role": "user", "content": prompt})

        # Get action thought
        action_thought = await self.llm.get_action_thought(self.chat_history)

        if not action_thought.is_action_needed:
            raise Exception("No action needed")

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

        if not action:
            raise Exception("Failed to create valid action")
        # Execute action and return results
        # TODO: Implement action execution
        return action

    async def execute(self, prompt: str, max_retries: int = 3) -> str:
        """Full execution pipeline"""
        action_data = await self.retrieve_or_generate(prompt, max_retries)
        execution_payload = await self.build_action_execution_payload(action_data)
        await self.execute_action(execution_payload)
        summary = await self.summarize_execution()
        return summary
