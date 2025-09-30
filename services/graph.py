from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from typing import Annotated
from .agents import router_agent, search_agent, summarizer_agent, routing_logic

class State(TypedDict):
    messages: Annotated[list, "add_messages"]
    answer: str


def build_graph():
    graph = StateGraph(State)

    # Add nodes
    graph.add_node("router_agent", router_agent)
    graph.add_node("search_agent", search_agent)
    graph.add_node("summarizer_agent", summarizer_agent)

    # Start to Router
    graph.add_edge(START, "router_agent")

    # Router decides first step
    graph.add_conditional_edges(
        "router_agent",
        routing_logic,
        {
            "summarizer_agent": "summarizer_agent",
            "search_agent": "search_agent",
        },
    )

    # After search optionally summarize
    def post_search_routing(state):
        query_text = state["messages"][0] if state["messages"] else ""
        if any(word in query_text.lower() for word in ["summarize", "summary", "summarise"]):
            return "summarizer_agent"
        return END

    graph.add_conditional_edges(
        "search_agent",
        post_search_routing,
        {
            "summarizer_agent": "summarizer_agent",
            END: END,
        },
    )

    # Summarizer to End
    graph.add_edge("summarizer_agent", END)

    return graph.compile()
