


analyst_pronmpt = """You are QA analyst and based on the feature provided,  
provide the test scerios in short for another qa writer so that he can write test cases based on that
Below is the feature
{feature}"""

writer_promt = """You are QA test case writer, Please write the test cases in short based on the these scenrioes. Below is the scenario -
{scenrio}"""


from typing import TypedDict

class AgentClass(TypedDict):
    feature: str
    scenrio: str
    test_cases:str

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

llm = OllamaLLM(model = "llama3.2")

analyst_node = PromptTemplate.from_template(analyst_pronmpt) | llm
writer_node = PromptTemplate.from_template(writer_promt) | llm

def anlyst_Agent(state: AgentClass):
    scenrio = analyst_node.invoke({"feature": state["feature"]})
    # print (scenrio)
    return {"scenrio": scenrio}


def writer_agent(state:AgentClass):
    writer_response = writer_node.invoke({"scenrio" : state["scenrio"]})
    # print(writer_response)
    return {"test_cases": writer_response}


# result = anlyst_Agent({"feature" : "A380", "scenrio":"", "test_cases":""})
# writer_agent({"feature" : "", "scenrio" : result, "test_cases" : ""})


# print(analyst_node)

from langgraph.graph import StateGraph, END

graph = StateGraph(AgentClass)
graph.add_node("analyser", anlyst_Agent)
graph.add_node("writer", writer_agent)


graph.set_entry_point("analyser")
graph.add_edge("analyser", "writer")
graph.add_edge("writer",END)

app = graph.compile()
result = app.invoke({"feature":"A380","scenrio":"","test_cases":""})
print(result["test_cases"])



from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM

class OllamaModel(DeepEvalBaseLLM):
    def __init__(self):
        self.model = OllamaLLM(model="llama3.2")

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt)

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "llama3.2"

ollama_model = OllamaModel()

test_case = LLMTestCase(
    input="A380",
    actual_output=result["writer_response"]
)

metric = AnswerRelevancyMetric(model=ollama_model)
metric.measure(test_case)

print("Score:", metric.score)
print("Reason:", metric.reason)



               





