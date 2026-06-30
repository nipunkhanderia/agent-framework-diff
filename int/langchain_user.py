from langchain_ollama import OllamaLLM

llm = OllamaLLM(model = "llama3.2")


# response = llm.invoke("why is sky blue, tell me in short")
# print(response)


from langchain_ollama import ChatOllama

llm_chat = ChatOllama(model = "llama3.2")
# llama_response = llm_chat.invoke("What is ocean? tell me in short")
# print(llama_response.content)

analyst_prompt = """You are an test scenrio analyst - Read the feature description and generate the the analyst test scenrios that can then
be passed on the writer prompt to generate test cases. Write it in short. Balow is the feature
---{feature}
"""

writer_prompt =""""You are QA writer. Based on the test scenrios identifed by analyst prompt, please write the test cases
for those scenarios as qa writer, read the description below.Write it in short.
These are the scenrios and these are the format of steps, build fon it
{scenrios}
Name - TC -1
Steps
expected result """

from langchain_core.prompts import PromptTemplate


analyst_agent = PromptTemplate.from_template(analyst_prompt) | llm
writer_agent = PromptTemplate.from_template(writer_prompt) | llm

# print(analyst_agent)
feature = "A380"

def get_scenrios(feature):
    scenrios = analyst_agent.invoke({"feature": feature})
    # print (scenrios)
    return scenrios
def get_test(scenrios):
    tests = writer_agent.invoke({"scenrios":scenrios})
    # print(tests)
    return tests

scenrios = get_scenrios(feature)
test_cases = get_test(scenrios)

print(scenrios)
print(test_cases)



