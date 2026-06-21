"""
SIMPLE CrewAI Example
=======================
Same 2-agent task as the LangGraph example:

  1. Analyst   -> reads a feature description, writes test scenarios
  2. Writer    -> reads those scenarios, writes detailed test cases

The difference from LangGraph/LangChain: CrewAI hands work between agents
AUTOMATICALLY. You don't build a State dictionary or call .invoke() with
filled-in blanks yourself - you just describe WHO each agent is (Agent)
and WHAT needs doing (Task), then tell CrewAI the order. CrewAI does the
hand-off for you.

Before running, install the package this file needs:
    pip install crewai

And make sure Ollama is running with the model pulled:
    ollama pull llama3.2
    ollama serve
"""

from crewai import Agent, Task, Crew, Process, LLM


# ===========================================================================
# STEP 1: Connect to the model
# ===========================================================================
# CrewAI talks to Ollama using the naming pattern "ollama/<model name>".
llm = LLM(model="ollama/llama3.2", base_url="http://localhost:11434")


# ===========================================================================
# STEP 2: The input - the feature we want test cases for
# ===========================================================================
SAMPLE_FEATURE = """
Feature: User Login
As a registered user, I want to log in with my email and password
so that I can access my account dashboard.

Acceptance Criteria:
- User can log in with a valid email and password   
- User sees an error message for invalid credentials
- Account locks after 5 failed login attempts
"""


# ===========================================================================
# STEP 3: Agent 1 - the Analyst
# ===========================================================================
# An Agent is just a description of WHO is doing the work: a role, a goal,
# and a backstory (extra context that shapes how it behaves). No actual
# instructions go here - that's what Task is for, below.
analyst = Agent(
    role="QA Requirements Analyst",
    goal="Extract clear test scenarios from a feature description",
    backstory="You are meticulous about covering positive, negative, and edge cases.",
    llm=llm,
    verbose=True,
)


# ===========================================================================
# STEP 4: Agent 2 - the Test Writer
# ===========================================================================
writer = Agent(
    role="QA Test Case Writer",
    goal="Turn test scenarios into detailed, step-by-step test cases",
    backstory="You write test cases any QA engineer could execute without ambiguity.",
    llm=llm,
    verbose=True,
)


# ===========================================================================
# STEP 5: Task 1 - what the Analyst must actually do
# ===========================================================================
# A Task is the real instruction, plus who is responsible for it.
# Notice the feature text is just dropped straight into the description
# with an f-string - no {placeholder} blanks, no .invoke() needed here.
analyze_task = Task(
    description=f"""Read this feature description and list test scenarios
(positive, negative, edge cases) as a numbered list:

{SAMPLE_FEATURE}""",
    expected_output="A numbered list of test scenarios.",
    agent=analyst,
)


# ===========================================================================
# STEP 6: Task 2 - what the Writer must actually do
# ===========================================================================
# context=[analyze_task] is the important line. It tells CrewAI:
# "before running this task, run analyze_task first, and automatically
#  hand its output to this task." That's the ENTIRE hand-off - compare
# this to manually writing state["scenarios"] in the LangGraph version.
write_task = Task(
    description="""Using the scenarios identified by the Analyst, write
detailed test cases. Each test case needs: Test ID, Title, Steps,
Expected Result.""",
    expected_output="A list of formatted test cases.",
    agent=writer,
    context=[analyze_task],
)


# ===========================================================================
# STEP 7: Crew - group the agents and tasks together, and set the order
# ===========================================================================
crew = Crew(
    agents=[analyst, writer],
    tasks=[analyze_task, write_task],
    process=Process.sequential,  # run analyze_task first, then write_task
    verbose=True,
)


# ===========================================================================
# STEP 8: Run it
# ===========================================================================
if __name__ == "__main__":
    result = crew.kickoff()
    print("\n=== FINAL RESULT ===")
    print(result)