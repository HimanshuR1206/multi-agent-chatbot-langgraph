import random
from urllib.parse import quote
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import config
from prompts.prompts import IMAGE_GEN_PROMPT
from utils.llm import get_llm


def image_gen_node(state):
    request = state["messages"][-1].content
    prompt = get_llm(config.CHAT_MODEL, 0.7).invoke(
        [SystemMessage(content=IMAGE_GEN_PROMPT), HumanMessage(content=request)]
    ).content.strip()

    # Pollinations generates the image from the URL itself (free, no key).
    url = (
        f"https://image.pollinations.ai/prompt/{quote(prompt)}"
        f"?width=1024&height=1024&nologo=true&seed={random.randint(1, 10**6)}"
    )
    return {
        "messages": [AIMessage(content=f"Generated image for: *{prompt}*")],
        "output_images": [url],
    }
