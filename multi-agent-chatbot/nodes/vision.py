from langchain_core.messages import HumanMessage
import config
from utils.llm import get_llm


def vision_node(state):
    text = state["messages"][-1].content or "Describe this image in detail."
    mime = state.get("image_mime") or "image/jpeg"
    data_uri = f"data:{mime};base64,{state['image_b64']}"
    msg = HumanMessage(content=[
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": data_uri}},
    ])
    resp = get_llm(config.VISION_MODEL, 0.2).invoke([msg])
    return {"messages": [resp]}
