from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class ActionCollectiveRequest(BaseModel):
    """Discover actions when an action or tool is needed but no provided action is sufficient."""

    thought: str = Field(
        ...,
        description="Thoughts about the task that needs to be solved and how to resolve it.",
    )
    tool_description: str = Field(
        ..., description="Description of the tool that can be used to resolve the task."
    )


class ActionDataGenerator(BaseModel):
    """Action data for general use of action"""

    input_json_schema: str = Field(
        ...,
        description="""JSON Schema defining the expected input parameters for the action function.
Simple example of a JSON Schema for input_json_schema and output_json_schema:
```
{
    "type": "object",
    "description": "The description of this item",
    "properties": {
        "id": {
            "description": "The id of this inner item",
            "type": "integer"
        },
        "value": {
            "type": "array",
            "description": "The list of values of this inner item",
            "items": {
                "type": "string",
                "description": "The value of this inner item",
                "enum": ["a", "b"]
            },
        }
    },
    "required": ["value", "id"],
    "additionalProperties": false,
}```
MAKE SURE TO INCLUDE A DESCRIPTION FOR ALL PROPERTIES INCLUDING THE ROOT OBJECT AND ALL NESTED PROPERTIES
MAKE SURE TO ADD THE REQUIRED ITEM""",
    )
    output_json_schema: str = Field(
        ...,
        description="JSON Schema defining the expected return value from the action function. FOLLOW THE EXAMPLE DELIMITED BY TRIPLE BACKTICKS",
    )
    code: str = Field(
        ...,
        description="""Python code implementing a function named 'action' that takes parameters matching the input_json_schema.
It must return a value matching the output_json_schema.
You only have access to the following libraries: numpy, requests. You must import them at the top of the file if you would like to use them.
Example delimited by triple backticks:
```
def action(input1: int, input2: int) -> dict:
    # sum of the two inputs

    return {"result": input1 + input2}
```""",
    )
    test: str = Field(
        ...,
        description="""This is code that will be appended to the previous code. It will be used to test the action function.
It must use assertions to validate the action function works correctly with static test inputs derived from the current task.
The test should NEVER be quantitative, it should only be qualitative. If it does need to be quantitative than it should be code not hard values.
Example delimited by triple backticks:
```
i1 = 7
i2 = 8
sum = action(input1, input2)
assert isinstance(sum["result"], int)
assert sum["result"] == i1 + i2
```""",
    )
