# from typing import TypedDict
# from langchain_core.prompts import PromptTemplate
# from langchain_ollama import OllamaLLM
# import langchain

# langchain.debug = True


# ANALYST_PROMPT = """You are a QA Requirements Analyst.
# Read the feature description below and write a numbered list of test
# scenarios (positive, negative, and edge cases).

# Feature description:
# {feature}
# """

# llm = OllamaLLM(model="llama3.2", base_url="http://localhost:11434")

# class AgentState(TypedDict):
#     feature:str
#     scenario:str
#     test_cases:str

# def anlayst_node(state:AgentState):
#     prompt = PromptTemplate.from_template(ANALYST_PROMPT)
#     print(prompt.format(feature=state["feature"]))
#     print("=" * 50)
#     chain = prompt | llm
#     # print(chain)
#     scenarios = chain.invoke({"feature":state["feature"]})
#     print(scenarios)


# anlayst_node({"feature":"Airbus A380", "scenario":"2121","test_cases":"21212"})


from typing import TypedDict
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

llm = OllamaLLM(model="llama3.2", base_url="http://localhost:11434")

class AgentState(TypedDict):
    feature: str
    scenarios: str
    usecase: str

Analyst_prompt = """You are a QA Requirements Analyst.
Read the feature description below and write a numbered list of test
scenarios (positive, negative, and edge cases).

Feature description:
{feature}
"""

writer_prompt = """You are a QA Testcase writer.
Read the feature description below and write test cases.

Test ID: TC-XX
Title: <short title>
Steps:
 1. ...
Expected Result: ...

Scenarios:
{scenarios}
"""

state = AgentState()
def analyst_node(state):
    prompt = PromptTemplate.from_template(Analyst_prompt)
    chain = prompt | llm
    scenarios = chain.invoke({"feature":state["feature"]})
    # print(scenarios)
    return {"scenarios": scenarios}



def writer_node(state):
    prompt = PromptTemplate.from_template(writer_prompt)
    chain = prompt | llm
    usecase = chain.invoke({"scenarios" : state["scenarios"]})
    # print(usecase)
    return{"usecase":usecase}


# analyst_result = analyst_node({"feature" : "Airbus A380", "scenarios":"", "usecase" : ""})
# writer_node({"feature":"", "scenarios":analyst_result["scenarios"], "usecase":""})

from langgraph.graph import StateGraph, END
graph = StateGraph(AgentState)

graph.add_node("analyst",analyst_node)
graph.add_node("writer", writer_node)


graph.set_entry_point("analyst")
graph.add_edge("analyst","writer")
graph.add_edge("writer", END)


app = graph.compile()


if __name__ == "__main__":
    starting_state = {
        "feature":"Airbus A380",
        "scenario":"",
        "usecase":""
    }
    result = app.invoke(starting_state)
    print("===================================Below are the use cases==================================")
    print(result["usecase"])