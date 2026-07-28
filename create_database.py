from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv
import os
import tempfile
import uuid
import time

# Load environment variables from a local .env file (used when running locally).
load_dotenv()

# Pointer file that always records the path of the CURRENT active Chroma
# database. Each rebuild creates a brand-new, uniquely named database
# directory instead of deleting/reusing the same one — this avoids
# "readonly database" errors that occurred when repeatedly deleting and
# recreating the same fixed path in a cloud container's filesystem.
CHROMA_POINTER = os.path.join(tempfile.gettempdir(), "docreader_chroma_pointer.txt")

# Where uploaded documents are stored before processing.
# Uses the system temp directory so this works both locally and on
# Streamlit Cloud, where the app's own source folder is read-only.
DATA_PATH = os.path.join(tempfile.gettempdir(), "docreader_data")


def main():
    generate_data_store()


def generate_data_store():
    """
    Full ingestion pipeline: load documents, split into chunks,
    embed and save to a new Chroma database.
    """
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    """Loads all markdown files from DATA_PATH into LangChain Document objects."""
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    """
    Splits documents into small, overlapping chunks. Overlap helps
    preserve context that spans chunk boundaries.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # Sanity-check print of the first chunk (safe even for very short
    # documents, unlike the original hardcoded chunks[10] which broke
    # on short uploads).
    if chunks:
        print(chunks[0].page_content)
        print(chunks[0].metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    """
    Embeds and saves chunks to a NEW, uniquely named Chroma database
    directory (rather than overwriting a fixed path), then updates the
    pointer file so query_data.py knows which database is current.

    Embedding is done in small batches with pauses between them to stay
    within Gemini's free-tier rate limit (100 requests/minute).
    """
    new_chroma_path = os.path.join(tempfile.gettempdir(), f"docreader_chroma_{uuid.uuid4().hex}")

    embedding_function = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    db = Chroma(persist_directory=new_chroma_path, embedding_function=embedding_function)

    batch_size = 25
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        db.add_documents(batch)
        print(f"Embedded batch {i // batch_size + 1} ({i + len(batch)}/{len(chunks)} chunks)")
        time.sleep(30)  # stay safely under 100 requests/minute

    # Record this new database as the current active one.
    with open(CHROMA_POINTER, "w") as f:
        f.write(new_chroma_path)

    print(f"Saved {len(chunks)} chunks to {new_chroma_path}.")


if __name__ == "__main__":
    main()