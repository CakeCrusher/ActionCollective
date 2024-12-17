import os
import json
import asyncio

from pydantic import BaseModel, Field
from action_collective import ActionClient, ActionCollectiveRequest
from dotenv import load_dotenv
from serpapi import GoogleSearch
import openai

# Load environment variables
load_dotenv()


class WebBrowserToolInput(BaseModel):
    """Browse the web for live data"""

    q: str = Field(..., description="The query to search for")


class SearchWebWrapper(BaseModel):
    """Searches google"""

    api_key: str = Field(
        os.getenv("SERPAPI_API_KEY", ""),
        description="The api key to use for the search",
    )

    def raw_results(self, query: str) -> str:
        search = GoogleSearch(
            {"q": query, "location": "China", "api_key": os.getenv("SERPAPI_API_KEY")}
        )
        results = search.get_dict()
        organic_results = results.get("organic_results", [])
        formatted_results = []
        for results in organic_results[:3]:
            title = results.get("title", "")
            snippet = results.get("snippet", "")
            formatted_results.append(f"Title: {title}\nSnippet: {snippet}\n")

        print(formatted_results)
        return "\n".join(formatted_results)


def validate_summary(summary: str) -> None:
    matrix = [
        [215, 230, 227, 260, 275],
        [479, 518, 515, 596, 635],
        [765, 830, 817, 960, 1025],
        [919, 998, 1035, 1156, 1235],
        [1315, 1430, 1407, 1660, 1775],
    ]
    # validate that each of the number inside matrix exist in the result string
    for row in matrix:
        for number in row:
            assert str(number) in summary
    print("\n\nPASSED\n\n")


async def main():
    """Test matrix multiplication action"""
    openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    action_client = ActionClient(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        backend_url=os.getenv("BACKEND_URL", "http://localhost:8000"),
        verbose=True,
    )
    search_web_wrapper = SearchWebWrapper(api_key=os.getenv("SERPAPI_API_KEY", ""))
    prompt = """Please perform the matrix multiplication of A x B and return the result, here are the variables:
    A = [[1, 2, 3, 4, 5],
            [6, 7, 7, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 7, 19, 20],
            [21, 22, 23, 24, 25]]
    B = [[1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 7, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25]]"""
    # prompt = "Hi!"
    # prompt = "What is the stock price for Amazon?"

    chat_history = [{"role": "user", "content": prompt}]

    web_browser_tool = openai.pydantic_function_tool(WebBrowserToolInput)
    action_collective_tool = openai.pydantic_function_tool(ActionCollectiveRequest)

    res = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=chat_history,
        tools=[
            web_browser_tool,
            action_collective_tool,
        ],
    )

    print("INITIAL:\n", res.model_dump_json(indent=4))

    if tool_calls := res.choices[0].message.tool_calls:
        chat_history.append({
            "role": "assistant",
            "content": res.choices[0].message.content,
            "tool_calls": [tool_call.model_dump() for tool_call in tool_calls]
        })
        for tool_call in tool_calls:
            tool_message = {"role": "tool", "content": "", "tool_call_id": tool_call.id}
            tool_args = json.loads(tool_call.function.arguments)
            if tool_call.function.name == web_browser_tool["function"]["name"]:
                formatted_output = {"result": search_web_wrapper.raw_results(tool_args["q"])}
                str_output = json.dumps(formatted_output)
                tool_message["content"] = str_output
            elif tool_call.function.name == action_collective_tool["function"]["name"]:
                # set coonfigs
                action_client.chat_history = chat_history[:-1]
                action_client.action_thought = ActionCollectiveRequest(**tool_args)
                # execute only needed steps
                await action_client.retrieve_or_generate()
                await action_client.build_action_execution_payload()
                await action_client.execute_action()

                formatted_output = {"result": action_client.result}
                str_output = json.dumps(formatted_output)
                tool_message["content"] = str_output
            chat_history.append(tool_message)

    summary_res = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=chat_history,
    )

    print("SUMMARY:\n", summary_res.model_dump_json(indent=4))

    # result = await client.execute(chat_history=chat_history)
    # print("\n\nresult:\n", json.dumps(result, indent=4), "\n\n")
    # validate_summary(result["content"])


if __name__ == "__main__":
    asyncio.run(main())
