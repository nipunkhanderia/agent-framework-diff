"""
LangChain version
==================
LangChain has NO built-in concept of "agents talking to each other".
To make Agent 1 hand off to Agent 2, YOU manually take the output of the
first chain and feed it into the second one. That manual wiring is the
whole point of comparing this against LangGraph and CrewAI below.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from langchain_ollama import OllamaLLM  # noqa: E402
from langchain_core.prompts import PromptTemplate  # noqa: E402

from shared.config import (  # noqa: E402
    MODEL_NAME,
    OLLAMA_BASE_URL,
    SAMPLE_FEATURE,
    ANALYST_PROMPT,
    WRITER_PROMPT,
)


def main():
    llm = OllamaLLM(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)

    # --- Agent 1: Analyst ---
    print("=== Agent 1 (Analyst) is working ===")
    analyst_chain = PromptTemplate.from_template(ANALYST_PROMPT) | llm
    scenarios = analyst_chain.invoke({"feature": SAMPLE_FEATURE})
    print(scenarios)

    # --- Manual hand-off: we pass Agent 1's output into Agent 2 ourselves ---
    print("\n=== Agent 2 (Test Writer) is working ===")
    writer_chain = PromptTemplate.from_template(WRITER_PROMPT) | llm
    test_cases = writer_chain.invoke({"scenarios": scenarios})
    print(test_cases)

    return test_cases


if __name__ == "__main__":
    main()
