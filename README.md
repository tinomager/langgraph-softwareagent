# LangGraphAgent

## Overview

LangGraphAgent is a project designed to create a ReAct agent that solves complex software problems. For the beginning, the problem is stated as a fixed input and the agent will try to solve the problem with step by step reasoning and the Tavily websearch tool to research the solution. If the Langsmith Environment variables are set, the traces will be sent to Langgraph for tracing the agent.

The project uses Azure OpenAI Service as the LLM (GPT4o and GPT4o-mini).

## Files

### `langgraphagent.py`

This is the main script of the project. It sets up the environment, defines the agent's behavior, and executes the plan. Key components include:

- **Environment Setup**: Loads environment variables and configures logging.
- **System Prompt**: Defines the initial prompt for the agent.
- **LLM and Tools**: Configures the language model and tools used by the agent.
- **Plan and Execution**: Defines the plan, execution steps, and replanning logic.
- **Workflow**: Sets up the state graph for the agent's workflow.
- **Main Function**: Executes the main logic of the agent.

The agent implements the following workflow:

![Graph](/graph.png)

### `.env`

This file contains the environment variables required for the project. These include API keys, endpoints, and configuration settings.
See sample.env for the required configuration values:

- **AZURE_OPENAI_API_KEY** for the API key of your Azure OpenAI Service deployment
- **AZURE_OPENAI_ENDPOINT** for the API endpoint of the Azure OpenAI Service deployment
- **OPENAI_API_VERSION** for the API version to be used
- **TAVILY_API_KEY** for an API key for the Tavily search as a LLM tool
- **GRAPH_RENDER_FILE** a file name the graph is rendered if not already existing
- **LANGSMITH_TRACING** set this if you want to trace the agent to langsmith, if not set, only local logging is used 
- **LANGSMITH_ENDPOINT** set this if you want to trace the agent to langsmith, if not set, only local logging is used
- **LANGSMITH_API_KEY** set this if you want to trace the agent to langsmith, if not set, only local logging is used
- **LANGSMITH_PROJECT** set this if you want to trace the agent to langsmith, if not set, only local logging is 
- **GPT_4O_MODEL** for the name of the GPT4o model deployment in Azure
- **GPT_4O_MINI_MODEL** for the name of the GPT4o-mini model deployment in Azure

## Getting Started

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/langgraphagent.git
    cd langgraphagent
    ```

2. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Set up environment variables**:
    - Create a `.env` file in the root directory.
    - Copy the contents from the provided `.env` example and update the values as needed.

4. **Run the agent**:
    ```sh
    python langgraphagent.py
    ```

## License

This project is licensed under the MIT License.