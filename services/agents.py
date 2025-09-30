from langgraph.prebuilt import create_react_agent
from .tools import web_search, summarizer_text
from .model import llm

def _normalize_messages(messages):
    normalized = []
    for m in messages:
        if isinstance(m, dict) and "content" in m:
            normalized.append(m)
        elif isinstance(m, str):
            normalized.append({"role": "user", "content": m})
        else:
            normalized.append({"role": "user", "content": str(m)})
    return normalized

def search_agent(state):
    """Agent that performs web search."""
    agent = create_react_agent(llm, [web_search])
    messages = _normalize_messages(state["messages"])
    result = agent.invoke({"messages": messages})
    last_msg = result["messages"][-1]
    output = getattr(last_msg, "content", str(last_msg))
    state["messages"].append(output)
    state["answer"] = output
    return state

def summarizer_agent(state):
    """Agent that summarizes the original user query."""
    
    text_to_summarize = state["messages"][0] if state["messages"] else ""

    if not text_to_summarize.strip():
        summary = "No text provided to summarize."
    else:
        summary = summarizer_text(text_to_summarize)

    state["messages"].append(summary)
    state["answer"] = summary
    return state

def router_agent(state):
    """Router agent decides which specialized agent to use first."""
    state["messages"].append("Router received the query and will decide next agent.")
    return state

def routing_logic(state):
    """
    Decide which agent to call: search_agent or summarizer_agent.
    """
    query_text = state["messages"][0] if state["messages"] else ""

    # Check Keyword first
    if any(word in query_text.lower() for word in ["summarize", "summary", "summarise"]):
        return "summarizer_agent"

    # To LLM
    prompt = (
        "You are a short intent classifier. Respond with exactly one word: "
        "'search', 'summarize', or 'search_and_summarize'.\n"
        f"Query: {query_text}\n\nAnswer:"
    )
    try:
        resp = llm.invoke(prompt)
        decision = resp.content.strip().lower()
        if "summarize" in decision and "search" not in decision:
            return "summarizer_agent"
        if "search_and_summarize" in decision or ("search" in decision and "summarize" in decision):
            return "search_agent"
        if "search" in decision:
            return "search_agent"
    except Exception:
        pass

    return "search_agent"
