"""
SIMPLE CrewAI Example - with a quality-check / retry step
============================================================
Based on your LangGraph version that uses `check_scenario_quality` to decide
whether to retry the Analyst or move on to the Writer.

CrewAI's equivalent of that custom conditional-edge logic is a Task
"guardrail" - a function attached directly to a Task that checks its
output. If the guardrail says the output isn't good enough, CrewAI
automatically re-runs that same task (showing the agent your error message
as feedback), up to `guardrail_max_retries` times.

IMPORTANT DIFFERENCE from your LangGraph version - read this before you run it:
  - Your LangGraph code, after retries > 3, just GIVES UP and moves on to
    the writer anyway with whatever scenarios it has.
  - CrewAI's guardrail, after running out of retries, RAISES AN ERROR and
    stops the whole crew instead of moving on. There's no built-in
    "give up gracefully and continue anyway" behavior here - if you want
    that, you have to catch the error yourself (see STEP 9 at the bottom).

Install:
    pip install crewai
Make sure Ollama is running with the model pulled:
    ollama pull llama3.2
    ollama serve
"""

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tasks.task_output import TaskOutput


# ===========================================================================
# STEP 1: Connect to the model
# ===========================================================================
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
# STEP 5: The quality check - this IS your check_scenario_quality function,
# just rewritten in the shape CrewAI expects (called a "guardrail")
# ===========================================================================
# A guardrail function always:
#   - receives the task's output (a TaskOutput object - .raw is the text)
#   - returns a tuple:
#       (True, output)            -> good enough, let it through
#       (False, "error message")  -> not good enough, please retry
def check_scenario_quality(output: TaskOutput):
    response_length = output.raw.count("\n")
    print("Response length is " + str(response_length))

    if response_length > 3:
        return (True, output.raw)

    # Not enough detail. CrewAI will retry the task and show the agent
    # this exact message as feedback on what to fix.
    return (
        False,
        "The scenario list is too short. Please cover positive, negative, "
        "AND edge cases in more detail.",
    )


# ===========================================================================
# STEP 6: Task 1 - the Analyst, with the guardrail attached
# ===========================================================================
analyze_task = Task(
    description=f"""Read this feature description and list test scenarios
(positive, negative, edge cases) as a numbered list:

{SAMPLE_FEATURE}""",
    expected_output="A numbered list of test scenarios.",
    agent=analyst,
    guardrail=check_scenario_quality,
    guardrail_max_retries=3,  # matches the "retries > 3" cutoff you used in LangGraph
)


# ===========================================================================
# STEP 7: Task 2 - the Writer, automatically fed Task 1's output
# ===========================================================================
write_task = Task(
    description="""Using the scenarios identified by the Analyst, write
detailed test cases. Each test case needs: Test ID, Title, Steps,
Expected Result.""",
    expected_output="A list of formatted test cases.",
    agent=writer,
    context=[analyze_task],
)


# ===========================================================================
# STEP 8: Crew
# ===========================================================================
crew = Crew(
    agents=[analyst, writer],
    tasks=[analyze_task, write_task],
    process=Process.sequential,
    verbose=True,
)


# ===========================================================================
# STEP 9: Run it
# ===========================================================================
if __name__ == "__main__":
    try:
        result = crew.kickoff()
        print("\n=== FINAL RESULT (test cases) ===")
        print(result)
    except Exception as e:
        # This is where the behavior differs from your LangGraph version.
        # LangGraph would have moved on to the writer anyway after 3
        # retries; CrewAI's guardrail stops the crew here instead.
        print("\nThe Analyst couldn't produce a good-enough scenario list")
        print("after several retries, so CrewAI stopped the crew:")
        print(e)