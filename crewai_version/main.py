"""
CrewAI version
==============
CrewAI is BUILT around multiple agents working together. The Writer's Task
declares context=[analyze_task], so it automatically receives the Analyst's
output -- no manual wiring, no graph to draw. This is the most "agents
talking to each other out of the box" of the three frameworks.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from crewai import Agent, Task, Crew, Process, LLM  # noqa: E402

from shared.config import MODEL_NAME, OLLAMA_BASE_URL, SAMPLE_FEATURE  # noqa: E402

# CrewAI talks to Ollama through LiteLLM's "ollama/<model>" naming convention.
ollama_llm = LLM(model=f"ollama/{MODEL_NAME}", base_url=OLLAMA_BASE_URL)

analyst = Agent(
    role="QA Requirements Analyst",
    goal="Extract clear test scenarios from a feature description",
    backstory="You are meticulous about covering positive, negative, and edge cases.",
    llm=ollama_llm,
    verbose=True,
)

writer = Agent(
    role="QA Test Case Writer",
    goal="Turn test scenarios into detailed, step-by-step test cases",
    backstory="You write test cases any QA engineer could execute without ambiguity.",
    llm=ollama_llm,
    verbose=True,
)

analyze_task = Task(
    description=(
        "Read this feature description and list test scenarios "
        "(positive, negative, edge cases) as a numbered list:\n\n"
        f"{SAMPLE_FEATURE}"
    ),
    expected_output="A numbered list of test scenarios.",
    agent=analyst,
)

write_task = Task(
    description=(
        "Using the scenarios identified by the Analyst, write detailed test "
        "cases. Each test case needs: Test ID, Title, Steps, Expected Result."
    ),
    expected_output="A list of formatted test cases.",
    agent=writer,
    context=[analyze_task],  # <-- this is the entire "hand-off", built in
)

crew = Crew(
    agents=[analyst, writer],
    tasks=[analyze_task, write_task],
    process=Process.sequential,
    verbose=True,
)


def main():
    result = crew.kickoff()
    print(result)
    return result


if __name__ == "__main__":
    main()
