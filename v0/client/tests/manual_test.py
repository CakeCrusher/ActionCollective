import os
import asyncio
from action_collective import ActionClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


async def main():
    """Test matrix multiplication action"""
    client = ActionClient(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        backend_url=os.getenv("BACKEND_URL", "http://localhost:8000"),
        verbose=True,
    )
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

    result = await client.execute(prompt)
    print("\n\nresult:\n", result, "\n\n")
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
            assert str(number) in result
    assert result is not None
    assert isinstance(result, str)
    print("\n\nPASSED\n\n")


if __name__ == "__main__":
    asyncio.run(main())
