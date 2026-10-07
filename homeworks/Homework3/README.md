# Homework 3 — ReAct Agent with RAG and Python REPL Tools

A LangChain ReAct agent, backed by a local Ollama model, that can answer
questions either by reasoning directly, running Python code, searching a
document database, or chaining both together.

## What it does

`app.py` starts an interactive command-line session. For each question you
ask, the agent decides step by step which tool (if any) it needs, calls it,
reads the result, and repeats until it can give a final answer. Every tool
call and its result is printed as it happens, so you can see the agent's
reasoning unfold.

## Tools

- **`Python_REPL`** — the built-in LangChain `PythonREPLTool`. Executes
  arbitrary Python and returns whatever is printed. Required by this
  assignment; not wrapped or modified.
- **`rag_lookup`** — a custom tool (`rag_tool.py`) that searches a vector
  database built from the document corpus loaded in
  [Homework2](../Homework2). Returns the most relevant chunks, each tagged
  with its source.

### Why `rag_lookup` doesn't query Homework2's database directly

Homework2's Chroma store was built with Vertex AI's `gemini-embedding-001`
embeddings (3072 dimensions). This assignment requires Ollama's
`nomic-embed-text` embeddings (768 dimensions) instead, and a vector index
can't be searched with query vectors of a different dimension than the ones
it was built with.

So `rag_lookup` reads Homework2's documents and metadata with a single
read-only `collection.get()` call — no vectors computed, nothing written
back, Homework2's database and source files are never modified — and
re-embeds that same text with `nomic-embed-text` into a brand-new Chroma
collection under `Homework3/rag_data/`, built once on first run and reused
after that.

## Setup

1. Install dependencies:

   ```bash
   uv sync
   ```

2. Copy `.env.example` to `.env` and fill in your Ollama server details:

   ```bash
   cp .env.example .env
   ```

   | Variable | Meaning | Default |
   |---|---|---|
   | `OLLAMA_BASE_URL` | URL of your Ollama server | `http://localhost:11434` |
   | `OLLAMA_CHAT_MODEL` | Chat model for the agent | `llama3:8b` |
   | `OLLAMA_EMBED_MODEL` | Embedding model for `rag_lookup` | `nomic-embed-text` |

   Both models must already be pulled on that Ollama server.

3. Make sure Homework2's `rag_data/.chromadb` exists and has been populated
   (`cd ../Homework2 && uv run app.py --load`), since `rag_lookup` reads
   from it the first time it runs.

## Running it

```bash
uv run app.py
```

Ask a question at the `agent>>` prompt, or press Enter on a blank line to
quit. The first question that uses `rag_lookup` will take longer than usual,
since that's when the Homework2 corpus gets re-embedded locally.

### Example questions

- *"What does the RAG database say about how to create a list in Python?"*
  — answered from the document corpus alone.
- *"What is 23 times 19?"* — answered with the Python REPL alone.
- *"According to the RAG database, how do you get the last item of a list?
  Then use Python to demonstrate it on [10, 20, 30, 40]."* — uses both
  tools in sequence.
