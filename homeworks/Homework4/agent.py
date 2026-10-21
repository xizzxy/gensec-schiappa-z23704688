"""Interactive FastAgent that coordinates a Notes MCP server and a Git MCP server.

The agent connects to two MCP servers defined in ``fastagent.config.yaml``:

* ``notes`` -- the custom FastMCP server in ``mcp-server.py`` (add_note,
  list_notes, get_note, delete_note).
* ``git`` -- the official ``mcp-server-git`` package, pointed at this
  course repository, used to inspect commit history, diffs and status.

Safety constraint: the agent may only use READ-ONLY git tools. Write tools
(git_commit, git_add, git_reset, git_checkout, git_create_branch) are
blocked twice: the instruction forbids them, and the ``tools`` filter below
never exposes them to the model at all.

No API key is stored in this file. The model provider key is read by
fast-agent from the ``ANTHROPIC_API_KEY`` environment variable.

Usage (from homeworks/Homework4):
    uv run agent.py
"""

import asyncio
from pathlib import Path

from fast_agent.core.fastagent import FastAgent

# Course repository root. This is the same directory fastagent.config.yaml
# passes to mcp-server-git as ``--repository ../..`` (relative to this
# folder); the git server rejects any other repo_path.
REPO_PATH = Path(__file__).resolve().parents[2]

# Only these git tools are ever offered to the model. Anything that can
# modify or destroy repository state is deliberately absent.
READ_ONLY_GIT_TOOLS = [
    "git_status",
    "git_log",
    "git_show",
    "git_diff",
    "git_diff_staged",
    "git_diff_unstaged",
    "git_branch",
]

INSTRUCTION = f"""You are a helpful assistant for a student's course repository.

You have two sets of tools:

1. Git tools (server "git") to inspect this repository: commit history,
   individual commits, diffs, branches and working-tree status.
2. Notes tools (server "notes") to manage short text notes:
   add_note, list_notes, get_note and delete_note.

REPOSITORY PATH:
- Every git tool call (git_log, git_status, git_branch, git_show, git_diff,
  etc.) must pass exactly this repo_path:
  {REPO_PATH}
- Do not guess or try other paths such as "." or a subfolder; the git
  server only accepts this one repository.

GIT SAFETY RULES -- these are mandatory:
- Only use READ-ONLY git tools: git_log, git_show, git_diff,
  git_diff_staged, git_diff_unstaged, git_status and git_branch.
- NEVER call git_commit, git_add, git_reset, git_checkout or
  git_create_branch. They can modify or destroy repository history.
- If the user asks you to change the repository, refuse and explain that
  you only have read access.

WORKFLOW FOR "SUMMARIZE RECENT WORK" REQUESTS:
1. Call git_log to read the recent commit history.
2. Use git_show or git_diff on interesting commits if you need detail.
3. Write a concise summary of what changed and why.
4. Save that summary with add_note, using a descriptive title, and tell the
   user the ID of the note you created.

Be concise, and base answers on tool results rather than guesses.
"""

fast = FastAgent("Homework 4 Agent")


@fast.agent(
    instruction=INSTRUCTION,
    model="haiku",
    servers=["notes", "git"],
    tools={"git": READ_ONLY_GIT_TOOLS},
    use_history=True,
)
async def main() -> None:
    """Start the agent and open an interactive chat session.

    Launches the ``notes`` and ``git`` MCP servers over stdio (as configured
    in ``fastagent.config.yaml``), exposes the notes tools plus only the
    read-only git tools to the model, and hands control to fast-agent's
    interactive prompt until the user exits.
    """
    async with fast.run() as agent:
        await agent.interactive()


if __name__ == "__main__":
    asyncio.run(main())
