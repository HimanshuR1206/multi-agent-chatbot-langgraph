from langchain_core.messages import HumanMessage, SystemMessage
import config
from prompts.prompts import ROUTER_PROMPT
from utils.llm import get_llm

VALID = {"chat", "rag", "web", "image_search", "image_gen"}


def router_node(state):
    # An attached image always goes to the vision agent.
    if state.get("image_b64"):
        return {"route": "vision"}

    question = state["messages"][-1].content
    resp = get_llm(config.ROUTER_MODEL, 0).invoke(
        [SystemMessage(content=ROUTER_PROMPT), HumanMessage(content=question)]
    )
    words = resp.content.strip().lower().split()
    label = words[0].strip(".,:;'\"`") if words else "chat"
    return {"route": label if label in VALID else "chat"}


def pick_route(state) -> str:
    return state["route"]