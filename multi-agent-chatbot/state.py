from typing import Annotated, List, Optional, TypedDict
from langgraph.graph.message import add_messages


class ChatState(TypedDict, total=False):
    messages: Annotated[list, add_messages]  # conversation (kept by the checkpointer)
    route: str                               # which agent the router picked
    image_b64: Optional[str]                 # uploaded image for the vision agent
    image_mime: Optional[str]
    output_images: List[str]                 # image URLs to display in the UI
