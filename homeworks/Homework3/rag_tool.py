"""Custom RAG retrieval tool backed by the Homework2 document corpus.

Homework2 built its Chroma vector store using Vertex AI's
gemini-embedding-001 embeddings (3072 dimensions). This assignment requires
Ollama's nomic-embed-text embeddings (768 dimensions) instead, and the two
embedding spaces are incompatible: Chroma's vector index cannot be searched
with query vectors of a different dimension than the ones it was built with.

To reuse Homework2's corpus without ever modifying it, this module opens
Homework2's Chroma database in a strictly read-only fashion (a single
``collection.get()`` call to pull out the already-chunked document text and
metadata), then re-embeds that same text with nomic-embed-text into a new,
separate Chroma collection that belongs to Homework3. Homework2's database
and source files are never written to.
"""

import os

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings

HOMEWORK2_CHROMA_PATH = os.getenv(
    "HOMEWORK2_CHROMA_PATH", "../Homework2/rag_data/.chromadb"
)
HOMEWORK2_COLLECTION_NAME = os.getenv("HOMEWORK2_COLLECTION_NAME", "langchain")
RAG_CHROMA_PATH = os.getenv("RAG_CHROMA_PATH", "./rag_data/.chromadb")
RAG_COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "homework3_rag")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))

_vectorstore = None


def _read_homework2_documents():
    """Read every chunk's text and metadata out of Homework2's Chroma DB.

    Opens Homework2's persisted Chroma database directly through the
    chromadb client (bypassing LangChain's Chroma wrapper, which would
    require supplying an embedding function) and issues a single read-only
    ``collection.get()`` call. No vectors are computed and nothing is
    written back, so Homework2's database is left completely untouched.

    Returns:
        list[langchain_core.documents.Document]: The source chunks, with
        their original metadata preserved.
    """
    client = chromadb.PersistentClient(path=HOMEWORK2_CHROMA_PATH)
    collection = client.get_collection(HOMEWORK2_COLLECTION_NAME)
    result = collection.get(include=["documents", "metadatas"])

    documents = []
    for text, metadata in zip(result["documents"], result["metadatas"]):
        documents.append(Document(page_content=text, metadata=metadata or {}))
    return documents


def _build_embeddings():
    """Create the Ollama embedding function used for this assignment.

    Returns:
        langchain_ollama.OllamaEmbeddings: Configured to call the
        nomic-embed-text model on the configured Ollama server.
    """
    return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)


def get_or_build_vectorstore():
    """Open Homework3's Chroma store, populating it on first run only.

    Reuses a module-level instance across calls within the same process.
    If Homework3's local Chroma collection is empty, this reads Homework2's
    corpus read-only via :func:`_read_homework2_documents` and embeds it
    once with nomic-embed-text, so this and future runs can search it
    semantically without re-embedding every time.

    Returns:
        langchain_chroma.Chroma: The Homework3 vector store, ready to be
        turned into a retriever.
    """
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    vectorstore = Chroma(
        collection_name=RAG_COLLECTION_NAME,
        embedding_function=_build_embeddings(),
        persist_directory=RAG_CHROMA_PATH,
    )

    if len(vectorstore.get()["ids"]) == 0:
        documents = _read_homework2_documents()
        if documents:
            vectorstore.add_documents(documents=documents)

    _vectorstore = vectorstore
    return _vectorstore


def format_retrieved_docs(docs):
    """Format retrieved document chunks into a single tool-result string.

    Args:
        docs (list[langchain_core.documents.Document]): Chunks returned by
            the retriever, most relevant first.

    Returns:
        str: Each chunk's source and text, separated for readability, or a
        message stating that nothing relevant was found.
    """
    if not docs:
        return "No relevant documents were found in the RAG database."

    blocks = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        blocks.append(f"[source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


@tool
def rag_lookup(query: str) -> str:
    """Search the RAG document database for passages relevant to a query.

    Use this tool to answer questions about the course corpus loaded in
    Homework2 (LangChain documentation/Wikipedia pages, the course GitHub
    example file, the course YouTube transcript, and any locally loaded
    txt/pdf/docx/md/csv files). Do not use it for general knowledge
    questions or for anything requiring computation; use the Python REPL
    tool for that instead.

    Args:
        query: A natural-language question or topic to search for.

    Returns:
        The most relevant document chunks found, each tagged with its
        source, or a message saying nothing relevant was found.
    """
    vectorstore = get_or_build_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": RAG_TOP_K})
    docs = retriever.invoke(query)
    return format_retrieved_docs(docs)
