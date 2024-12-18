from typing import List
from pydantic import BaseModel, Field
import json
import openai
import os
import random
import dotenv

dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = openai.OpenAI(api_key=OPENAI_API_KEY)


class TaskGenerate(BaseModel):
    """Individual self encompassing task"""

    description: str = Field(..., description="The description of the task")
    independent: bool = Field(
        ..., description="Whether the task is independent of other tasks"
    )
    self_contained: bool = Field(
        ...,
        description="""Self-Contained Reasoning and Retrieval from Prompt/Web
Tasks that can be solved using the information provided directly in the prompt or found easily via general web search, without requiring interaction with external files, specialized code execution, or offline data sources.""",
    )


class TasksGenerate(BaseModel):
    """List of tasks"""

    tasks: List[TaskGenerate]


class Task(TaskGenerate):
    id: str = Field(..., description="The unique identifier for the task")


def steps_str_to_tasks(steps_str: str) -> List[Task]:
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": steps_str}],
        response_format=TasksGenerate,
    )
    generated_tasks = completion.choices[0].message.parsed

    if generated_tasks is None or generated_tasks.tasks is None:
        raise ValueError("No tasks generated")

    assigned_ids = [
        Task(id=str(random.randint(0, 1000000)), **task.model_dump())
        for task in generated_tasks.tasks
    ]
    return assigned_ids


def get_tasks(file_path: str, output_file_path: str):
    # get tasks from metadata.jsonl
    total_tasks = 0
    independent_and_self_contained_count = 0
    line_count = 0
    with open(file_path, "r") as f:
        for line in f:
            line_count += 1
            metadata = json.loads(line)
            if "Annotator Metadata" not in metadata or "Steps" not in metadata["Annotator Metadata"]:
                print(f"Skipping line {line_count} because it doesn't have the required metadata")
                continue
            steps_str = metadata["Annotator Metadata"]["Steps"]
            tasks = steps_str_to_tasks(steps_str)
            # save to tasks.jsonl
            with open(output_file_path, "a") as f:
                for task in tasks:
                    f.write(task.model_dump_json() + "\n")
                    if task.self_contained and task.independent:
                        independent_and_self_contained_count += 1
            total_tasks += len(tasks)
            print(f"Processed line {line_count} \t|\t Total tasks: {total_tasks} \t|\t Independent and self contained: {independent_and_self_contained_count}")


if __name__ == "__main__":
    get_tasks("metadata.jsonl", "tasks.jsonl")
