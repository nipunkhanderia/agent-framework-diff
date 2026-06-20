"""
SIMPLE LangGraph Example
=========================
Two "agents" (just plain Python functions) pass information through a
2-step pipeline:

  1. Analyst   -> reads a feature description, writes test scenarios
  2. Writer    -> reads those scenarios, writes detailed test cases

Everything you need is in THIS ONE FILE. Read it top to bottom.

Before running, install the 3 packages this file needs:
    pip install langgraph langchain-ollama langchain-core

And make sure Ollama is running with the model pulled:
    ollama pull llama3.2
    ollama serve
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate


# ===========================================================================
# STEP 1: Connect to the model
# ===========================================================================
# This one object knows how to send text to llama3.2 running on your machine.
llm = OllamaLLM(model="llama3.2", base_url="http://localhost:11434")


# ===========================================================================
# STEP 2: The input, and instructions for each agent
# ===========================================================================
# This is the feature description both agents will work on.
SAMPLE_FEATURE = """
Feature: User Login
As a registered user, I want to log in with my email and password
so that I can access my account dashboard.

Acceptance Criteria:
- User can log in with a valid email and password
- User sees an error message for invalid credentials
- Account locks after 5 failed login attempts
"""

# The {feature} part is a blank that gets filled in later.
ANALYST_PROMPT = """You are a QA Requirements Analyst.
Read the feature description below and write a numbered list of test
scenarios (positive, negative, and edge cases).

Feature description:
{feature}
"""

# The {scenarios} part is a blank that gets filled in later.
WRITER_PROMPT = """You are a QA Test Case Writer.
Turn each scenario below into a detailed test case using this format:

Test ID: TC-XX
Title: <short title>
Steps:
 1. ...
Expected Result: ...

Scenarios:
{scenarios}
"""


# ===========================================================================
# STEP 3: The State - a shared dictionary passed between agents
# ===========================================================================
# Think of this as a form with 3 blanks. Each agent fills in one more blank.
class AgentState(TypedDict):
    feature: str       # the input - filled in before we start
    scenarios: str      # filled in by the Analyst
    test_cases: str     # filled in by the Writer


# ===========================================================================
# STEP 4: Agent 1 - the Analyst
# ===========================================================================
def analyst_node(state: AgentState) -> dict:
    print("=== Analyst is working ===")

    # Take the prompt template, fill in {feature}, send it to the model.
    prompt = PromptTemplate.from_template(ANALYST_PROMPT)
    chain = prompt | llm
    scenarios = chain.invoke({"feature": state["feature"]})

    print(scenarios)

    # Only return the NEW piece of info. LangGraph merges it into the
    # shared state for us automatically.
    return {"scenarios": scenarios}


# ===========================================================================
# STEP 5: Agent 2 - the Test Writer
# ===========================================================================
def writer_node(state: AgentState) -> dict:
    print("\n=== Writer is working ===")

    prompt = PromptTemplate.from_template(WRITER_PROMPT)
    chain = prompt | llm
    test_cases = chain.invoke({"scenarios": state["scenarios"]})

    print(test_cases)

    return {"test_cases": test_cases}


# ===========================================================================
# STEP 6: Wire the two agents together as a graph
# ===========================================================================
graph = StateGraph(AgentState)

graph.add_node("analyst", analyst_node)     # register agent 1
graph.add_node("writer", writer_node)       # register agent 2

graph.set_entry_point("analyst")            # start here
graph.add_edge("analyst", "writer")         # analyst hands off to writer
graph.add_edge("writer", END)               # writer is the last step

app = graph.compile()                       # turn the description into
                                             # something we can actually run


# ===========================================================================
# STEP 7: Run it
# ===========================================================================
if __name__ == "__main__":
    starting_state = {
        "feature": SAMPLE_FEATURE,
        "scenarios": "",
        "test_cases": "",
    }

    result = app.invoke(starting_state)

    print("\n=== FINAL RESULT ===")
    print(result["test_cases"])