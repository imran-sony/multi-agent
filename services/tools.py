from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain.schema import HumanMessage, SystemMessage
import os
from .model import llm

tavily_api_key = os.getenv("TAVILY_API_KEY")


@tool("web_search")
def web_search(query: str) -> str:
    """
    Perform a real-time web search.
    """
    try:
        search = TavilySearch(max_results=3, tavily_api_key=tavily_api_key)
        results = search.invoke(query)
        formatted_results = "\n".join(
            [f"- {r['title']}: {r['content'][:1000]}..." for r in results]
        )
        return formatted_results if results else "No results found."
    except Exception as e:
        return f"Search error: {str(e)}"


def summarizer_text(query: str) -> str:
    """
    Summarize text using ChatGroq LLM with proper BaseMessage input.
    """
    if not query.strip():
        return "No text provided to summarize."

    messages = [
        SystemMessage(content="You are a helpful assistant that summarizes text in 5 sentences."),
        HumanMessage(content=query)
    ]

    try:
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        return f"Summary error: {str(e)}"
