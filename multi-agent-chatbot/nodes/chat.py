from langchain_core.messages import SystemMessage
import config
from prompts.prompts import CHAT_SYSTEM
from utils.llm import get_llm


def chat_node(state):
    history = state["messages"][-config.HISTORY_WINDOW:]
    resp = get_llm().invoke([SystemMessage(content=CHAT_SYSTEM)] + history)
    return {"messages": [resp]}
