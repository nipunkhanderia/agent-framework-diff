

from crewai import LLM, Agent, Task, Crew, Process

llm = LLM(model = "ollama/llama3.2", base_url="http://localhost:11434") 

SAMPLE_FEATURE = """
Feature: User Login
As a registered user, I want to log in with my email and password
so that I can access my account dashboard.

Acceptance Criteria:
- User can log in with a valid email and password                           
- User sees an error message for invalid credentials
- Account locks after 5 failed login attempts
"""

qa_agent = Agent(
    role = "Qa Analyst",
    goal = "Identifyong the Qa points in the scenario",
    backstory = "We are a firm which are going to deploy an application and we need to make sure that the application does not have any defect and that why QA analyst need to identify test cases",
    llm = llm,
    verbose = True
)

writer_agent = Agent(
    role = "QA writer",
    goal = "Write the test cases based on the qa points identifed in an scenario",
    backstory = "We are a firm which are going to deploy an application and we need to make sure that the application does not have any defect and that why QA write needs to write test cases based on the Qa points identifed by qa anlyst",
    llm = llm,
    verbose = True
)


analyse_task = Task(
    description=f"""Read this feature description and list test scenarios
(positive, negative, edge cases) as a numbered list:

{SAMPLE_FEATURE}""",
    expected_output="A numbered list of test scenarios",
    agent= qa_agent

)

writer_task = Task(
    description="""Using the scenarios identified by the Analyst, write
detailed test cases. Each test case needs: Test ID, Title, Steps,
Expected Result.""",
    expected_output="A list of formatted test cases",
    agent = writer_agent,
    context= [analyse_task]
)


crew = Crew(
    agents = [qa_agent, writer_agent],
    tasks=[analyse_task, writer_task],
    process = Process.sequential,
    verbose = True
)



if __name__ == "__main__":
    result = crew.kickoff()
    print (result)



