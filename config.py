"""Central configuration. Keys come from .env locally or st.secrets on Streamlit Cloud."""
import os
from dotenv import load_dotenv

load_dotenv()


def _get(key, default=None):
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)


GROQ_API_KEY = _get("GROQ_API_KEY")

# Check https://console.groq.com/docs/models if a model name stops working.
CHAT_MODEL = "openai/gpt-oss-120b"
ROUTER_MODEL = "openai/gpt-oss-20b"
VISION_MODEL = "qwen/qwen3.8-27b"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DATA_DIR = "data"
FAISS_DIR = "faiss_index"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4
HISTORY_WINDOW = 12  # messages sent to the LLM
