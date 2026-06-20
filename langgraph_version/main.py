"""
LangGraph version
==================
LangGraph models the same two agents as NODES in a graph, connected by an
EDGE, sharing a typed STATE object. The hand-off is explicit and visual:
analyst -> writer -> END. This is the same idea as LangChain above, but the
flow is declared as a graph instead of glued together by hand.
"""
import os
import sys
from typing import TypedDict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from langgraph.graph import StateGraph, END  # noqa: E402
from langchain_ollama import OllamaLLM  # noqa: E402
from langchain_core.prompts import PromptTemplate  # noqa: E402

from shared.config import (  # noqa: E402
    MODEL_NAME,
    OLLAMA_BASE_URL,
    SAMPLE_FEATURE,
    ANALYST_PROMPT,
    WRITER_PROMPT,
)

llm = OllamaLLM(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)


class AgentState(TypedDict):
    feature: str
    scenarios: str
    test_cases: str


def analyst_node(state: AgentState) -> dict:
    print("=== Node 1 (Analyst) is working ===")
    chain = PromptTemplate.from_template(ANALYST_PROMPT) | llm
    scenarios = chain.invoke({"feature": state["feature"]})
    print(scenarios)
    return {"scenarios": scenarios}


def writer_node(state: AgentState) -> dict:
    print("\n=== Node 2 (Test Writer) is working ===")
    chain = PromptTemplate.from_template(WRITER_PROMPT) | llm
    test_cases = chain.invoke({"scenarios": state["scenarios"]})
    print(test_cases)
    return {"test_cases": test_cases}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("analyst", analyst_node)
    graph.add_node("writer", writer_node)
    graph.set_entry_point("analyst")
    graph.add_edge("analyst", "writer")
    graph.add_edge("writer", END)
    return graph.compile()


def main():
    app = build_graph()
    result = app.invoke({"feature": SAMPLE_FEATURE, "scenarios": "", "test_cases": ""})
    return result["test_cases"]


if __name__ == "__main__":
    main()
