# Multi-Agent Chatbot (LangGraph)

A router agent classifies each message and sends it to the right agent:

| Route | Agent | Tech |
|---|---|---|
| chat | General LLM | Groq Llama 3.3 70B |
| rag | PDF Q&A | FAISS + HuggingFace embeddings |
| web | Live web search | DuckDuckGo (`ddgs`) |
| vision | Image understanding | Groq Llama 4 Scout |
| image_search | Find photos | DuckDuckGo images |
| image_gen | Create images | Pollinations AI |

Conversation memory: LangGraph `MemorySaver` (per `thread_id`).

## Flow
```
START -> router -> (chat | rag | web | vision | image_search | image_gen) -> END
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add your GROQ_API_KEY (free at console.groq.com)
streamlit run app.py
```

Upload PDFs from the sidebar, or put them in `data/` and run `python ingest.py`.

## Try these
- "Explain decorators in Python" -> chat
- "Summarize the uploaded PDF" -> rag
- "Latest news on AI regulation" -> web
- "Show me images of the Taj Mahal" -> image_search
- "Generate an image of a robot reading a book" -> image_gen
- Attach an image and ask "what is in this?" -> vision

## Deploy
Push to GitHub -> Streamlit Community Cloud -> add `GROQ_API_KEY` under app secrets.
