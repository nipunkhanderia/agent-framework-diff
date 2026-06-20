from typing import TypedDict
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM


ANALYST_PROMPT = """You are a QA Requirements Analyst.
Read the feature description below and write a numbered list of test
scenarios (positive, negative, and edge cases).

Feature description:
{feature}
"""

llm = OllamaLLM(model="llama3.2", base_url="http://localhost:11434")

class AgentState(TypedDict):
    feature:str
    scenario:str
    test_cases:str

def anlayst_node(state:AgentState):
    prompt = PromptTemplate.from_template(ANALYST_PROMPT)
    print(prompt)
    chain = prompt | llm
    print(chain)
    scenarios = chain.invoke({})
    print(scenarios)


anlayst_node({"feature":"121", "scenario":"2121","test_cases":"21212"})