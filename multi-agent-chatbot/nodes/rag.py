from functools import lru_cache
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, SystemMessage
import config
from prompts.prompts import RAG_SYSTEM
from utils.embeddings import get_embeddings
from utils.llm import get_llm


@lru_cache(maxsize=1)
def get_vectorstore():
    """Load the FAISS index once. Call get_vectorstore.cache_clear() after re-ingesting."""
    if not (Path(config.FAISS_DIR) / "index.faiss").exists():
        return None
    return FAISS.load_local(
        config.FAISS_DIR, get_embeddings(), allow_dangerous_deserialization=True
    )


def rag_node(state):
    vs = get_vectorstore()
    if vs is None:
        return {"messages": [AIMessage(
            content="No documents are indexed yet. Upload a PDF in the sidebar "
                    "(or put PDFs in `data/` and run `python ingest.py`)."
        )]}

    question = state["messages"][-1].content
    docs = vs.similarity_search(question, k=config.TOP_K)
    context = "\n\n".join(
        f"(Source: {Path(d.metadata.get('source', 'doc')).name}, "
        f"page {int(d.metadata.get('page', 0)) + 1})\n{d.page_content}"
        for d in docs
    )
    history = state["messages"][-6:]
    resp = get_llm(temperature=0.1).invoke(
        [SystemMessage(content=RAG_SYSTEM.format(context=context))] + history
    )
    return {"messages": [resp]}
