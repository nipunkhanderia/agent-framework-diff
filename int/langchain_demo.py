from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

llm = OllamaLLM(model="llama3.2", base_url="http://localhost:11434")

analyst_prompt = """You are a QA Requirements Analyst.
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

analyst_chain = PromptTemplate.from_template(analyst_prompt) | llm
writer_chain = PromptTemplate.from_template(writer_prompt) | llm


def get_scenarios(feature: str) -> str:
    retries = 0
    while True:
        scenarios = analyst_chain.invoke({"feature": feature})
        retries += 1
        response_length = scenarios.count("\n")
        print("Response length is", response_length, "| retries:", retries)

        if response_length > 3 or retries > 3:
            return scenarios
        # otherwise loop again and retry the analyst call


def get_usecases(scenarios: str) -> str:
    return writer_chain.invoke({"scenarios": scenarios})


if __name__ == "__main__":
    feature = "Airbus A380"
    scenarios = get_scenarios(feature)
    usecases = get_usecases(scenarios)

    print("=================== Below are the use cases ===================")
    print(usecases)