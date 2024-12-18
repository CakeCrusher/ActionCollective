# Action Collective

A framework for dynamic action generation and reuse with LLMs.

## Installation

```bash
pip install action-collective
```

## Quick Start

```python
from action_collective import ActionClient
import json
import os

client = ActionClient(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    backend_url=os.getenv("BACKEND_URL", "http://70.179.0.242:11000"),
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

chat_history = [{"role": "user", "content": prompt}]

result = await client.execute(chat_history=chat_history)
print("\n\nresult:\n", json.dumps(result, indent=4), "\n\n")

# Validate the result
import numpy as np
A = [[1, 2, 3, 4, 5],
        [6, 7, 7, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 7, 19, 20],
        [21, 22, 23, 24, 25]]
B = [[1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 7, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25]]
matrix_result = np.dot(A, B)
# validate that each of the number inside matrix exist in the result string
assert result is not None
for row in matrix_result:
    for number in row:
        assert str(number) in result[-1]["content"]
print("\n\nPASSED\n\n")
```

## Features

- Dynamic action generation
- Action reuse through vector similarity
- Automatic validation and testing
- Easy integration with OpenAI models

## Documentation

For full documentation, visit our [GitHub repository](https://github.com/yourusername/ActionCollective).