import base64
import io
import uuid
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage
from PIL import Image

import config
from graph import build_graph
from ingest import build_index
from nodes.rag import get_vectorstore

st.set_page_config(page_title="Multi-Agent Chatbot", page_icon="🤖", layout="wide")

ROUTE_LABELS = {
    "chat": "💬 LLM", "rag": "📚 RAG", "web": "🌐 Web Search",
    "vision": "👁️ Vision", "image_search": "🖼️ Image Search", "image_gen": "🎨 Image Generation",
}


@st.cache_resource
def get_graph():
    return build_graph()


graph = get_graph()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("🤖 Multi-Agent Chatbot")
    if not config.GROQ_API_KEY:
        st.error("GROQ_API_KEY is missing. Add it to .env or Streamlit secrets.")

    st.subheader("📚 Documents (RAG)")
    pdfs = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Build index", disabled=not pdfs):
        with st.spinner("Indexing..."):
            Path(config.DATA_DIR).mkdir(exist_ok=True)
            paths = []
            for f in pdfs:
                p = Path(config.DATA_DIR) / f.name
                p.write_bytes(f.getvalue())
                paths.append(p)
            n = build_index(paths)
            get_vectorstore.cache_clear()
        st.success(f"Indexed {n} chunks.")

    st.subheader("👁️ Image (Vision)")
    img_file = st.file_uploader("Attach an image", type=["png", "jpg", "jpeg", "webp"])
    st.caption("While an image is attached, questions go to the vision agent. Remove it to go back to normal routing.")

    if st.button("🗑️ New chat"):
        st.session_state.history = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# ---------------- Chat history ----------------
st.header("Ask me anything")
for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        for url in m.get("images", []):
            st.image(url, width=300)
        if m.get("route"):
            st.caption(f"Answered by: {ROUTE_LABELS.get(m['route'], m['route'])}")


def encode_image(file) -> str:
    img = Image.open(file).convert("RGB")
    img.thumbnail((1024, 1024))  # keep payload small for the API
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------- Input ----------------
if prompt := st.chat_input("Chat, ask about your PDF, search the web, find or generate images..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        if img_file:
            st.image(img_file, width=250)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                result = graph.invoke(
                    {
                        "messages": [HumanMessage(content=prompt)],
                        "image_b64": encode_image(img_file) if img_file else None,
                        "image_mime": "image/jpeg",
                        "output_images": [],
                    },
                    config={"configurable": {"thread_id": st.session_state.thread_id}},
                )
            answer = result["messages"][-1].content
            images = result.get("output_images", [])
            route = result.get("route")

            st.markdown(answer)
            for url in images:
                st.image(url, width=300)
            st.caption(f"Answered by: {ROUTE_LABELS.get(route, route)}")
            st.session_state.history.append(
                {"role": "assistant", "content": answer, "images": images, "route": route}
            )
        except Exception as e:
            st.error(f"Something went wrong: {e}")
