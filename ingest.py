"""Build the FAISS index from PDFs.  CLI:  python ingest.py   (reads data/*.pdf)"""
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config
from utils.embeddings import get_embeddings


def build_index(pdf_paths) -> int:
    docs = []
    for p in pdf_paths:
        docs.extend(PyPDFLoader(str(p)).load())
    if not docs:
        raise ValueError("No readable pages found in the given PDFs.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    FAISS.from_documents(chunks, get_embeddings()).save_local(config.FAISS_DIR)
    return len(chunks)


if __name__ == "__main__":
    pdfs = sorted(Path(config.DATA_DIR).glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"Put at least one PDF in {config.DATA_DIR}/ first.")
    print(f"Indexed {build_index(pdfs)} chunks from {len(pdfs)} PDF(s).")
