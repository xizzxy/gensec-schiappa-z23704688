"""
DocuMind: A NotebookLM-style RAG application built on LangChain.

Combines document loading (from web, GitHub, YouTube, and local files) with
a retrieval-augmented question-answering chain, plus support for loading a
user-supplied PDF or URL at runtime.

Usage:
    uv run app.py --load          # build the vector database from the course corpus
    uv run app.py --query         # start an interactive Q&A session
    uv run app.py --add-pdf PATH  # (custom feature) add a PDF at runtime
    uv run app.py --add-url URL   # (custom feature) add a URL at runtime
"""

import os
import sys
import argparse

from dotenv import load_dotenv
load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    AsyncHtmlLoader, DirectoryLoader, TextLoader, PyPDFDirectoryLoader,
    PyPDFLoader, Docx2txtLoader, UnstructuredMarkdownLoader, ArxivLoader,
    CSVLoader, GithubFileLoader,
)
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_chroma import Chroma
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_vertexai import ChatVertexAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

from wikipedia_loader import WikipediaLoader

# ---------------------------------------------------------------------------
# Vector store setup
# ---------------------------------------------------------------------------

embedding_function = VertexAIEmbeddings(
    model_name="gemini-embedding-001",
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location="us-west1",
)

vectorstore = Chroma(
    embedding_function=embedding_function,
    persist_directory="./rag_data/.chromadb",
)


def load_docs(docs):
    """Split documents into chunks and store their embeddings in Chroma."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=10)
    splits = text_splitter.split_documents(docs)
    vectorstore.add_documents(documents=splits)


# ---------------------------------------------------------------------------
# Course-provided loaders
# ---------------------------------------------------------------------------

def load_urls(urls):
    """Load web pages from URLs and add them to the vector database."""
    load_docs(AsyncHtmlLoader(urls).load())


def load_wikipedia(query):
    """Load Wikipedia pages for a query and add them to the vector database."""
    docs = WikipediaLoader(query=query, load_max_docs=2).load()
    load_docs(docs)


def load_github(file):
    """Load matching GitHub repository files and add them to the database."""
    loader = GithubFileLoader(
        repo="wu4f/cs475-src",
        branch="main",
        github_api_url="https://api.github.com",
        file_filter=lambda file_path: file_path.endswith(file),
    )
    load_docs(loader.load())


def load_youtube(video_id):
    """Load a YouTube transcript and add it to the vector database."""
    transcript = YouTubeTranscriptApi().fetch(video_id)
    text = ' '.join(entry.text for entry in transcript)
    docs = [Document(page_content=text, metadata={'source': f'youtube:{video_id}'})]
    load_docs(docs)


def load_txt(directory):
    """Load text files from a directory into the vector database."""
    load_docs(DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader).load())


def load_pdf(directory):
    """Load PDF files from a directory into the vector database."""
    load_docs(PyPDFDirectoryLoader(directory).load())


def load_docx(directory):
    """Load DOCX files from a directory into the vector database."""
    load_docs(DirectoryLoader(directory, glob="**/*.docx", loader_cls=Docx2txtLoader).load())


def load_md(directory):
    """Load Markdown files from a directory into the vector database."""
    load_docs(DirectoryLoader(directory, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader).load())


def load_csv(directory):
    """Load CSV files from a directory into the vector database."""
    load_docs(DirectoryLoader(directory, glob="**/*.csv", loader_cls=CSVLoader).load())


# ---------------------------------------------------------------------------
# CUSTOM FEATURE: runtime PDF / URL loading (added for this homework)
# ---------------------------------------------------------------------------

def load_pdf_file(path):
    """Load a single user-specified PDF file at runtime and add it to the database.

    This is DocuMind's custom addition to the course RAG example: instead of
    only ingesting the fixed course corpus, a user can point the app at any
    PDF on disk and immediately query it.
    """
    docs = PyPDFLoader(path).load()
    for doc in docs:
        doc.metadata['source'] = f"user-pdf:{path}"
    load_docs(docs)


def load_url_runtime(url):
    """Load a single user-specified URL at runtime and add it to the database.

    Custom addition: lets a user add any web page as a new knowledge source
    without editing the script.
    """
    docs = AsyncHtmlLoader([url]).load()
    for doc in docs:
        doc.metadata['source'] = f"user-url:{url}"
    load_docs(docs)


# ---------------------------------------------------------------------------
# Corpus loading (course-provided baseline corpus)
# ---------------------------------------------------------------------------

def load_baseline_corpus():
    """Load the full course-provided document corpus into the vector database."""
    wiki_query = "LangChain"
    print(f"Loading Wikipedia pages on: {wiki_query}")
    load_wikipedia(wiki_query)

    github_file = "butcher.py"
    print(f"Loading github file(s) with ending: {github_file}")
    load_github(github_file)

    youtube_video_id = "78600iosmis"
    print(f"Loading YouTube video: {youtube_video_id}")
    load_youtube(youtube_video_id)

    print("Loading TXT files from: rag_data/txt")
    load_txt("rag_data/txt")

    print("Loading PDF files from: rag_data/pdf")
    load_pdf("rag_data/pdf")

    print("Loading DOCX files from: rag_data/docx")
    load_docx("rag_data/docx")

    print("Loading MD files from: rag_data/md")
    load_md("rag_data/md")

    print("Loading CSV files from: rag_data/csv")
    load_csv("rag_data/csv")

    print_sources()


def print_sources():
    """Print the unique source labels currently indexed in the vector database."""
    retriever = vectorstore.as_retriever()
    sources = set()
    for doc_metadata in retriever.vectorstore.get()['metadatas']:
        sources.add(doc_metadata['source'])
    print("Database sources:")
    for source in sources:
        print(f"  {source}")


# ---------------------------------------------------------------------------
# Query chain
# ---------------------------------------------------------------------------

def build_rag_chain():
    """Construct the retrieval-augmented generation chain used for Q&A."""
    llm = ChatVertexAI(model=os.getenv("GOOGLE_MODEL"), project=os.getenv("GOOGLE_CLOUD_PROJECT"), location="us-west1")
    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_template(
        """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:"""
    )

    def format_docs(docs):
        """Join retrieved document contents into one prompt context string."""
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def run_query_loop():
    """Start an interactive command-line question-answering session."""
    rag_chain = build_rag_chain()
    print("Welcome to DocuMind. Ask a question, or press Enter on a blank line to quit.")
    print_sources()
    while True:
        line = input("llm>> ")
        if not line:
            break
        print(rag_chain.invoke(line))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Parse command-line arguments and dispatch to the requested mode."""
    parser = argparse.ArgumentParser(description="DocuMind RAG application")
    parser.add_argument("--load", action="store_true", help="Load the baseline course corpus")
    parser.add_argument("--query", action="store_true", help="Start an interactive query session")
    parser.add_argument("--add-pdf", metavar="PATH", help="Add a PDF file to the database at runtime")
    parser.add_argument("--add-url", metavar="URL", help="Add a URL to the database at runtime")
    args = parser.parse_args()

    if args.load:
        load_baseline_corpus()
    if args.add_pdf:
        load_pdf_file(args.add_pdf)
        print(f"Added PDF: {args.add_pdf}")
    if args.add_url:
        load_url_runtime(args.add_url)
        print(f"Added URL: {args.add_url}")
    if args.query:
        run_query_loop()

    if not any([args.load, args.query, args.add_pdf, args.add_url]):
        parser.print_help()


if __name__ == "__main__":
    main()
