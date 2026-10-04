from functools import lru_cache
from langchain_groq import ChatGroq
import config


@lru_cache(maxsize=8)
def get_llm(model: str = None, temperature: float = 0.3) -> ChatGroq:
    return ChatGroq(
        model=model or config.CHAT_MODEL,
        temperature=temperature,
        api_key=config.GROQ_API_KEY,
    )
