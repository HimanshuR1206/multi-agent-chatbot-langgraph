from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import config
from prompts.prompts import IMAGE_QUERY_PROMPT
from utils.llm import get_llm
from utils.search import image_search


def image_search_node(state):
    request = state["messages"][-1].content
    query = get_llm(config.ROUTER_MODEL, 0).invoke(
        [SystemMessage(content=IMAGE_QUERY_PROMPT), HumanMessage(content=request)]
    ).content.strip()

    urls = [r["image"] for r in image_search(query) if r.get("image")]
    if not urls:
        return {"messages": [AIMessage(content=f"I couldn't find images for “{query}”.")], "output_images": []}
    return {
        "messages": [AIMessage(content=f"Here are some images for **{query}**:")],
        "output_images": urls,
    }
