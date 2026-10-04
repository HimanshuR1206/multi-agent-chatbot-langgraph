from datetime import date
from langchain_core.messages import AIMessage, SystemMessage
from prompts.prompts import WEB_SYSTEM
from utils.llm import get_llm
from utils.search import web_search


def web_node(state):
    question = state["messages"][-1].content
    results = web_search(question)
    if not results:
        return {"messages": [AIMessage(content="I couldn't fetch web results right now. Please try again.")]}

    context = "\n\n".join(
        f"[{i}] {r.get('title', '')}\n{r.get('body', '')}\nURL: {r.get('href', '')}"
        for i, r in enumerate(results, 1)
    )
    system = WEB_SYSTEM.format(today=date.today().isoformat(), context=context)
    resp = get_llm(temperature=0.2).invoke(
        [SystemMessage(content=system)] + state["messages"][-4:]
    )
    return {"messages": [resp]}
