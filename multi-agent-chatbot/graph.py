from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from nodes.chat import chat_node
from nodes.image_gen import image_gen_node
from nodes.image_search import image_search_node
from nodes.rag import rag_node
from nodes.router import pick_route, router_node
from nodes.vision import vision_node
from nodes.web_search import web_node
from state import ChatState


def build_graph():
    g = StateGraph(ChatState)

    g.add_node("router", router_node)
    g.add_node("chat", chat_node)
    g.add_node("rag", rag_node)
    g.add_node("web", web_node)
    g.add_node("vision", vision_node)
    g.add_node("image_search", image_search_node)
    g.add_node("image_gen", image_gen_node)

    g.add_edge(START, "router")
    g.add_conditional_edges(
        "router",
        pick_route,
        {
            "chat": "chat",
            "rag": "rag",
            "web": "web",
            "vision": "vision",
            "image_search": "image_search",
            "image_gen": "image_gen",
        },
    )
    for node in ["chat", "rag", "web", "vision", "image_search", "image_gen"]:
        g.add_edge(node, END)

    # MemorySaver = conversation memory per thread_id
    return g.compile(checkpointer=MemorySaver())
