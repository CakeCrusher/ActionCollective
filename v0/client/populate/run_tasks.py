import os
import json
import asyncio
import logging
from typing import List, Optional
from datetime import datetime

from pydantic import BaseModel, Field
from action_collective import ActionClient
from action_collective.models.actions import ActionData
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))


class TaskGenerate(BaseModel):
    """Individual self encompassing task"""
    description: str = Field(..., description="The description of the task")
    independent: bool = Field(..., description="Whether the task is independent of other tasks")
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


async def run_task(task: Task) -> Optional[ActionData]:
    try:
        action_client = ActionClient(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            backend_url=os.getenv("BACKEND_URL", "http://localhost:8000"),
            verbose=True,
        )
        chat_history = [{"role": "user", "content": task.description}]
        action_client.chat_history = chat_history
        await action_client.retrieve_or_generate(retrieve_threshold=1)
        return action_client.action_data
    except Exception as e:
        print(f"Error running task {task.id}: {e}")
        return None


async def save_action_data(action_data: ActionData, output_file: str):
    """Save action data to file with proper error handling"""
    try:
        # Load existing data or create new list
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                try:
                    action_datas = json.load(f)
                except json.JSONDecodeError:
                    action_datas = []
        else:
            action_datas = []

        # Append new action
        action_datas.append(action_data.model_dump())

        # Write back entire file
        with open(output_file, "w") as f:
            json.dump(action_datas, f)
        
        return True
    except Exception as e:
        print(f"Error saving action data: {e}")
        return False


async def setup_logging(output_file: str) -> logging.Logger:
    """Setup logging to both file and console"""
    # Create logger
    logger = logging.getLogger('action_collective')
    logger.setLevel(logging.INFO)
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s | %(message)s')
    console_formatter = logging.Formatter('%(message)s')
    
    # Create and configure file handler
    log_file = f"{output_file}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    
    # Create and configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # # Log initial message
    # logger.info(f"Starting action collection at {datetime.now().isoformat()}")
    
    return logger


async def main(input_file: str, output_file: str):
    # Setup logging
    logger = await setup_logging(output_file)
    
    task_count = 0
    self_contained_and_independent_count = 0
    
    with open(input_file, "r") as file:
        for line in file:
            task_count += 1
            task = Task.model_validate_json(line)
            
            if task.self_contained and task.independent:
                self_contained_and_independent_count += 1
                print(f"\nProcessing task {task.id}...")
                
                # Wait for task completion
                action_data = await run_task(task)
                
                if action_data:
                    # Wait for save completion
                    save_success = await save_action_data(action_data, output_file)
                    status = "completed successfully" if save_success else "completed but failed to save"
                else:
                    status = "FAILED"
                
                logger.info(f"Task {task.id} {status} | Total tasks: {task_count} | Self contained and independent tasks: {self_contained_and_independent_count}")
            else:
                logger.info(f"Task {task.id} SKIPPED | Total tasks: {task_count} | Self contained and independent tasks: {self_contained_and_independent_count}")
            
            # Add a small delay between tasks
            await asyncio.sleep(1)
    
    # logger.info(f"\nFinished action collection at {datetime.now().isoformat()}")


if __name__ == "__main__":
    asyncio.run(main("tasks.jsonl", "action_datas.json"))
