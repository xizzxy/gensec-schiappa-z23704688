"""
Chainlit web interface for DocuMind.

Wraps the RAG chain and custom PDF/URL loaders from app.py in a chat UI.
Users can ask questions, or type /addpdf <path> or /addurl <url> to add
a new source to the knowledge base without leaving the chat.
"""

import chainlit as cl

from app import build_rag_chain, load_pdf_file, load_url_runtime

rag_chain = build_rag_chain()


@cl.on_chat_start
async def on_chat_start():
    """Send the welcome message when a Chainlit session starts."""
    welcome_text = (
        "**Welcome to DocuMind!**\n\n"
        "Ask me anything from my set of documents, "
        "or type `/addpdf <path>` or `/addurl <url>` to add a new source.\n\n"
    )
    await cl.Message(content=welcome_text).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle a Chainlit user message: either add a source or answer a query."""
    text = message.content.strip()

    if text.startswith("/addpdf ") or text.startswith("/addurl "):
        command, _, arg = text.partition(" ")
        arg = arg.strip()
        if not arg:
            await cl.Message(content=f"Usage: {command} <value>").send()
            return
        try:
            if command == "/addpdf":
                load_pdf_file(arg)
                await cl.Message(content=f"Added PDF: {arg}").send()
            else:
                load_url_runtime(arg)
                await cl.Message(content=f"Added URL: {arg}").send()
        except Exception as e:
            await cl.Message(content=f"Failed to add source: {e}").send()
        return

    answer = rag_chain.invoke(text)
    await cl.Message(content=answer).send()


if __name__ == "__main__":
    cl.run()
