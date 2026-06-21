from typing import TypedDict
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
import langchain

langchain.debug = True


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
    print(prompt.format(feature=state["feature"]))
    print("=" * 50)
    chain = prompt | llm
    # print(chain)
    scenarios = chain.invoke({"feature":state["feature"]})
    print(scenarios)


anlayst_node({"feature":"Airbus A380", "scenario":"2121","test_cases":"21212"})