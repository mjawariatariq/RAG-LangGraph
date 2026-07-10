"""
Run once to build the FAISS vector index from sample_docs/*.txt
Usage: python ingest.py
"""

import glob
import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

DOCS_DIR = os.path.join(os.path.dirname(__file__), "sample_docs")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "faiss_index")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_and_split() -> list[Document]:
    """Load .txt files and split into ~500-char chunks with overlap."""
    chunks = []
    for path in glob.glob(os.path.join(DOCS_DIR, "*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        source = os.path.basename(path)
        # Split on double-newline (paragraph boundary)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        current = ""
        for para in paragraphs:
            if len(current) + len(para) + 2 <= CHUNK_SIZE:
                current = (current + "\n\n" + para).strip()
            else:
                if current:
                    chunks.append(Document(page_content=current, metadata={"source": source}))
                # If single paragraph is already > CHUNK_SIZE, keep it as-is
                current = para

        if current:
            chunks.append(Document(page_content=current, metadata={"source": source}))

    return chunks


def main():
    print("Loading and splitting documents...")
    chunks = load_and_split()
    print(f"  Split into {len(chunks)} chunks")

    print("Embedding and building FAISS index...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_DIR)

    print(f"Done! Index saved to {INDEX_DIR}/")
    print(f"  Total chunks indexed: {len(chunks)}")


if __name__ == "__main__":
    main()