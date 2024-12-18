# Action Collective
A framework for LLM agents that lets them dynamically generate and reuse actions by writing Python functions, rather than relying on a fixed set.

Inspired by [DynaSaur: Large Language Agents Beyond Predefined Actions](https://arxiv.org/abs/2411.01747)

Written by Sebastian Sosa in collaboration with [ChatGPT](https://chatgpt.com/share/675d01d9-82d4-8011-baf8-056340780afe) 

Presenting @ [Latent Space Discord](https://discord.gg/vGERHJVC) Paper Club 12/18

## Installation
```bash
pip install action-collective
```

## Basic Example
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

# Action Collective Manifesto: Community Driven Actions Database

## I. Introduction and Motivation

LLMs, trained on massive corpora, excel at logical deduction, problem-solving, and understanding complex domains. They can reason about when and how to use tools, making it possible to construct autonomous agents that solve problems via an unconstrained sequence of decisions—no longer limited to linear, predefined flows.

Traditional LLM-based agents rely on sets of hand-crafted tools to solve given tasks. While effective in controlled environments, this approach struggles in more freeform conditions requiring flexibility, creativity, and the discovery of new capabilities as challenges arise. The next logical step is to tap into the LLM’s own code generation prowess, allowing agents to dynamically produce, refine, and utilize new tools—actions represented as code. By doing so, we transform a static action space into one that can continuously expand and evolve.

This is where Action Collective comes in: a platform and ecosystem that empowers LLM agents to access a vast, ever-growing database of actions passively contributed by its users. As a result, agents gain the capacity to handle an increasingly diverse array of tasks, leveraging the collective's energy and ingenuity. Ultimately, this project transforms static tool use into a general-purpose, dynamic framework, supercharging the potential of LLM agents across myriad problem domains.

## II. The Core Objective

**Goal:** Elevate LLM agents from constrained problem-solvers to boundless solution-finders by enabling them to discover, create, and reuse dynamically generated actions.

**Vision:** Rather than manually coding every conceivable action, we will build an open ecosystem where users upload their own tested, reliable, and easily discoverable actions. By pooling the distributed cognitive work and computational effort of many contributors, our platform makes it effortless for LLM agents to tap into a library of powerful tools. Users gain access to a shared repository of capabilities. Not meerly sparing them the need to reinvent the wheel, but opening up a agentic paradigm where an agent can effortlessly be endowed with a vast array of capabilities.

The ultimate aspiration is for AI engineers to focus on the high-level logic and user experience of their agents, confident that a robust library of actions is there to support them. As a user, you spend less time painstakingly designing tools and more time innovating and refining your agent’s workflow.

## III. Architectural Overview

Our architecture is designed for security, accessibility, and scalability. Each component plays a unique role, ensuring seamless configurable integration between contributors, the action library, and LLM agents that utilize these actions.

### A. Authentication & User Management

Authentication ensures trust and accountability. Users authenticate through a secure, web-based process and receive API keys granting access to the ecosystem. This enables us to track contributions and usage, identify and remove malicious actors, and maintain a safe environment.

### B. Client SDK

The Client SDK is the linchpin that streamlines interaction with the entire platform. It allows users to:

- **Upload Actions:** Users submit actions—either generated by their LLM or manually written—directly through the SDK. The upload process consists of the following key features:

  - The python code representing the action.
  - JSON Schema defining an action, ensuring LLM compatibility.
  - For quality assurance a relevant and rigorous unit test is required.
  - Additional metadata is captured for attribution and retrieval.

- **Retrieve Actions:** On top of fine-grained filters, users and agents can easily find relevant actions. The SDK simplifies retrieval, ensuring users do not need to understand the underlying schema or indexing methods.

- **Feedback and Iteration:** Users passively report on the usefulness and reliability of actions. Over time, this feedback loop refines the ecosystem, surfacing the best solutions while weeding out less effective ones.

### C. Backend Services

The backend orchestrates business logic, ensuring that every action, query, and feedback loop adheres to our policies. It:

- **Manages Action Lifecycle:** Newly submitted actions pass through automated tests, scanning, and verification. Once validated, they are uploaded into the active database.
- **Integrates With Tests and Policies:** Automatic checks ensure no malicious code slips through. Policies are enforced seamlessly, so the user’s workflow remains uninterrupted.
- **Handles Load and Scalability:** With growing adoption, the backend is built to scale. Whether you are a single engineer testing your first contribution or a large team relying on the ecosystem at scale, performance remains smooth.

### D. Vector Database and Storage Layer

All actions are stored in a vector database, facilitating both semantic and traditional methods for retrieval. As the database grows, the LLM agent’s ability to find just the right action in a vast library remains quick and intuitive. This ensures that as new contributions flood in, the quality of results and retrieval speed remains consistent and reliable.

## IV. Security and Governance

Dynamic action generation and sharing raise security concerns, which we address proactively:

- **Authentication and Access Controls:** Strict user authentication via API keys lets us quickly respond to bad actors, removing or banning malicious users.
- **Automatic Policy Enforcement:** Both the client SDK and the backend run validations, ensuring no dangerous or malicious actions enter the system.
- **User-Level Filters:** Users can apply granular filters, retrieving only actions that align with their security, ethical, or domain-specific constraints.

Our governance approach balances openness with responsibility, ensuring that while the action space is unbounded, it remains safe and productive.

## V. Incentives and Ecosystem Growth

A thriving, community-driven ecosystem needs the right incentives. Here, we create a balanced “economy” of contributions:

- **Obfuscation for Fairness:** The system abstracts away whether an action comes from the user’s own set or the community. By blending sources, we promote mutual benefit, building trust that everyone is contributing for the common good.

- **Emergent Standards:** Over time, best practices and coding styles emerge organically. With the SDK guiding uploads and ensuring non-malicious code, users focus on producing high-quality, reusable actions that benefit the entire ecosystem.

## VI. Future Extensions and Opportunities

Looking ahead, we envision:

- **Automated Generalization:** Future iterations of the SDK and backend may assist in generalizing overly specific actions or merging similar ones, expanding utility and reducing duplication.
- **Seamless Integrations:** We plan to integrate with popular AI toolkits, such as the increasingly accepted OpenAI SDK, making it easy for users to plug into this platform no matter their preferred tooling.
- **Continuous Improvement:** Over time, the community and system co-evolve. As more actions pour in and more users participate, LLM agents become increasingly capable, creating a powerful feedback loop of innovation and problem-solving capacity.
- **Economy of Contributions and Credits:** We propose a balanced “give and take” model as the cornerstone of a thriving action ecosystem. Contributing high-quality actions earns users “credits,” which can then be spent to retrieve more actions. This fair exchange encourages a sustained flow of valuable contributions from the community, ensuring that all participants—both new and experienced—benefit from an ever-expanding library of capabilities.

## VII. Conclusion

We stand at the frontier of LLM-enabled autonomy. By transcending a fixed, predefined set of actions and embracing a dynamic, community-powered action repository, we empower AI engineers to scale their agents to new heights. Instead of painstakingly building and maintaining tools yourself, you tap into a collective intelligence—an ecosystem that continuously evolves and refines itself.

**Your Invitation:** We invite you to explore, contribute, and refine this evolving ecosystem. By sharing your actions and leveraging those of the community, you help shape a robust, secure, and open-source platform. As we collectively expand the action repository and strengthen its foundations, we enable LLM agents to become more adaptable, efficient, and capable than ever before.

This is an opportunity to work together in driving the next evolution of AI agents. Your expertise, insights, and input are valued—join us, and help build a future where AI tools are seamlessly integrated, always improving, and accessible to all.
