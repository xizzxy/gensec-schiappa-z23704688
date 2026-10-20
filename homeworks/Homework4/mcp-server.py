"""Simple local Notes MCP server built with FastMCP.

Notes are persisted to a ``notes.json`` file located in the same directory
as this script. No API keys are required.

Usage:
    python mcp-server.py            # stdio transport (default)
    python mcp-server.py stdio      # stdio transport
    python mcp-server.py http       # HTTP transport on 127.0.0.1:8000
"""

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastmcp import FastMCP

NOTES_FILE = Path(__file__).resolve().parent / "notes.json"
VALID_TRANSPORTS = ("stdio", "http")
HTTP_HOST = "127.0.0.1"
HTTP_PORT = 8000

mcp = FastMCP("Notes")


def load_notes() -> dict:
    """Load all notes from the local JSON file.

    Returns:
        dict: A mapping of note ID to note record. Returns an empty dict if
        the file does not exist or contains invalid JSON.
    """
    if not NOTES_FILE.exists():
        return {}
    try:
        with NOTES_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: could not read {NOTES_FILE}: {e}", file=sys.stderr)
        return {}
    return data if isinstance(data, dict) else {}


def save_notes(notes: dict) -> None:
    """Persist all notes to the local JSON file.

    Args:
        notes (dict): A mapping of note ID to note record.

    Returns:
        None
    """
    tmp_file = NOTES_FILE.with_suffix(".json.tmp")
    with tmp_file.open("w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2)
    tmp_file.replace(NOTES_FILE)


@mcp.tool()
def add_note(title: str, content: str) -> dict:
    """Create a new note and save it to disk.

    Args:
        title (str): The title of the note. Must not be empty.
        content (str): The body text of the note.

    Returns:
        dict: The created note record with keys ``id``, ``title``,
        ``content`` and ``created_at`` (ISO 8601 UTC timestamp), or an
        error dict if the title is empty.
    """
    if not title or not title.strip():
        return {"error": "Title must not be empty."}

    note = {
        "id": str(uuid.uuid4()),
        "title": title.strip(),
        "content": content,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    notes = load_notes()
    save_notes({**notes, note["id"]: note})
    return note


@mcp.tool()
def list_notes() -> list:
    """List all saved notes without their full content.

    Returns:
        list: A list of dicts, each with keys ``id``, ``title`` and
        ``created_at``, sorted by creation time (oldest first).
    """
    notes = load_notes()
    summaries = [
        {"id": n["id"], "title": n["title"], "created_at": n["created_at"]}
        for n in notes.values()
    ]
    return sorted(summaries, key=lambda n: n["created_at"])


@mcp.tool()
def get_note(note_id: str) -> dict:
    """Retrieve the full record for a single note.

    Args:
        note_id (str): The UUID of the note to retrieve.

    Returns:
        dict: The full note record (``id``, ``title``, ``content``,
        ``created_at``), or ``{"error": ...}`` if no note has that ID.
    """
    note = load_notes().get(note_id)
    if note is None:
        return {"error": f"Note with id '{note_id}' not found."}
    return note


@mcp.tool()
def delete_note(note_id: str) -> dict:
    """Delete a note by its ID.

    Args:
        note_id (str): The UUID of the note to delete.

    Returns:
        dict: ``{"status": "deleted", "id": note_id}`` on success, or
        ``{"error": ...}`` if no note has that ID.
    """
    notes = load_notes()
    if note_id not in notes:
        return {"error": f"Note with id '{note_id}' not found."}
    save_notes({k: v for k, v in notes.items() if k != note_id})
    return {"status": "deleted", "id": note_id}


def main() -> None:
    """Parse the transport argument and start the MCP server.

    Returns:
        None
    """
    transport = sys.argv[1].lower() if len(sys.argv) > 1 else "stdio"
    if transport not in VALID_TRANSPORTS:
        print(
            f"Usage: python {Path(__file__).name} [{'|'.join(VALID_TRANSPORTS)}]",
            file=sys.stderr,
        )
        sys.exit(1)

    if transport == "http":
        mcp.run(transport="http", host=HTTP_HOST, port=HTTP_PORT)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
