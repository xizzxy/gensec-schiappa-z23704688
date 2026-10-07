"""ReAct agent combining a Python REPL tool with a custom RAG retrieval tool.

Usage:
    uv run app.py

Starts an interactive command-line session. For each question, a ReAct
agent backed by a local Ollama chat model decides, step by step, whether to
answer directly, run Python code with the PythonREPLTool, search the
Homework2 document corpus with the custom rag_lookup tool (see
rag_tool.py), or some combination of both, before printing a final answer.
Each tool call and its result is streamed to the console as it happens.
Press Enter on a blank line to quit.
"""

import os

from dotenv import load_dotenv

load_dotenv()

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_experimental.tools import PythonREPLTool
from langchain_ollama import ChatOllama

from rag_tool import rag_lookup

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3:8b")

# Standard ReAct prompt structure (Yao et al., 2022), written out locally
# instead of pulled from the LangChain Hub so the agent has no runtime
# dependency on an external prompt registry. Includes one extra line
# reminding the model that the Python REPL tool only reports printed
# output, since the base ReAct format alone isn't enough for a local model
# to infer that on its own.
REACT_PROMPT_TEMPLATE = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

When using the Python_REPL tool, the Action Input must call print(...) around
any value you want to see, since only printed output becomes the Observation.

If the question references the RAG database or any documents, call
rag_lookup before doing anything else.

Every line that starts with "Thought:" must be immediately followed by
either an "Action:" line or a "Final Answer:" line -- never leave a
Thought without one of those two directly after it.

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


def build_agent_executor():
    """Construct the ReAct agent executor with its two tools.

    Returns:
        langchain_classic.agents.AgentExecutor: Runs a ReAct loop over an
        Ollama chat model (llama3:8b by default) with access to the
        PythonREPLTool (required by the assignment) and the custom
        rag_lookup tool from rag_tool.py.
    """
    llm = ChatOllama(model=OLLAMA_CHAT_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.3)
    tools = [PythonREPLTool(), rag_lookup]
    prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=10,
    )


def run_query_loop():
    """Start an interactive command-line session with the ReAct agent.

    Reads one question per line, streams each intermediate tool call and
    its observation to the console as the agent produces them, then prints
    the final answer. An empty line ends the session.
    """
    agent_executor = build_agent_executor()
    print("Welcome to the Homework3 ReAct agent.")
    print(f"Chat model: {OLLAMA_CHAT_MODEL}  |  Server: {OLLAMA_BASE_URL}")
    print("Tools available: Python REPL, RAG document lookup.")
    print("Ask a question, or press Enter on a blank line to quit.")

    while True:
        question = input("\nagent>> ")
        if not question:
            break

        for step in agent_executor.stream({"input": question}):
            if "actions" in step:
                for action in step["actions"]:
                    print(f"  [tool call] {action.tool}({action.tool_input!r})")
            if "steps" in step:
                for intermediate_step in step["steps"]:
                    print(f"  [tool result] {intermediate_step.observation}")
            if "output" in step:
                print(f"\nFinal Answer: {step['output']}")


def main():
    """Entry point: start the interactive ReAct agent session."""
    run_query_loop()


if __name__ == "__main__":
    main()
