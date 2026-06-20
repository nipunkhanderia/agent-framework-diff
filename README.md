# Agent Framework Diff: LangChain vs LangGraph vs CrewAI

A small, beginner-friendly project that builds the **exact same** two-agent
QA workflow three times — once in LangChain, once in LangGraph, once in
CrewAI — so you can see, in real code, how each framework handles multiple
agents talking to each other. All three run locally against **Ollama +
llama3.2**, and GitHub Actions runs lint/tests automatically, with an
optional one-click workflow that runs the real thing end-to-end.

## The task (identical in all three versions)

```
Feature description
        |
        v
+-------------------+        +----------------------+
| Agent 1: Analyst   | -----> | Agent 2: Test Writer |
| extracts test      |        | turns scenarios into |
| scenarios          |        | detailed test cases  |
+-------------------+        +----------------------+
```

Same model, same prompts, same input (`shared/config.py`) — only the
orchestration code differs.

## Folder structure

```
agent-framework-diff/
├── shared/config.py          # model name, prompts, sample input - shared by all 3
├── langchain_version/main.py # manual hand-off between two LangChain chains
├── langgraph_version/main.py # two nodes in a StateGraph, connected by an edge
├── crewai_version/main.py    # two CrewAI Agents/Tasks, hand-off via `context=`
├── tests/test_smoke.py       # fast tests that don't need a live LLM
├── conftest.py                # lets pytest find the folders above
├── requirements.txt
└── .github/workflows/
    ├── ci.yml                # lint + smoke tests, runs on every push
    └── e2e-ollama.yml        # installs Ollama, pulls llama3.2, runs all 3 for real (manual trigger)
```

## How the three frameworks differ here

| | LangChain | LangGraph | CrewAI |
|---|---|---|---|
| How Agent 1 hands off to Agent 2 | You do it manually — pass the first chain's output into the second chain's input | Declared as an edge in a graph (`add_edge("analyst", "writer")`) | Built in — Task 2 declares `context=[task_1]` |
| Core concept | Chains (`prompt \| llm`) | Nodes + shared State + Edges | Agents + Tasks + Crew |
| Best for | Simple, linear pipelines | Multi-step flows with branching/loops/visibility into state | Multi-agent "team" workflows out of the box |
| Code needed for this 2-agent task | Least ceremony, but you own the wiring | A bit more setup (define State, nodes, edges) | Most structure, but the hand-off is free |

## Prerequisites

1. **Python 3.10+**
2. **[Ollama](https://ollama.com)** installed locally
3. Pull the model once: `ollama pull llama3.2`

## Setup (step by step, beginner-friendly)

```bash
# 1. Clone your repo and go into it
git clone <your-repo-url>
cd agent-framework-diff

# 2. Create a virtual environment (keeps dependencies isolated)
python -m venv venv

# 3. Activate it
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. Make sure Ollama is running in another terminal
ollama serve

# 6. Run any version
python langchain_version/main.py
python langgraph_version/main.py
python crewai_version/main.py
```

Each script prints out what "Agent 1" and "Agent 2" produced, so you can
watch the hand-off happen.

## Running tests locally

```bash
pytest tests/ -v
```

These tests don't call the real model — they check the wiring (prompts,
graph, crew) is correct, so they run fast and don't need Ollama running.

## CI/CD

- **`ci.yml`** runs automatically on every push/PR: installs dependencies,
  lints with flake8, runs the smoke tests. Takes under a minute, doesn't
  need a model.
- **`e2e-ollama.yml`** is a manual workflow (Actions tab → "End-to-End Demo"
  → "Run workflow"). It installs Ollama on the runner, pulls llama3.2, and
  actually runs all three pipelines for real — a genuine end-to-end test,
  not a mock. It's manual-trigger because pulling a model takes a few
  minutes, and you don't want that on every commit.

## Ideas to extend this

- Add a 3rd agent ("Reviewer") that checks the test cases for completeness
- Swap `llama3.2` for another Ollama model and compare output quality
- Add a `requirements-lock.txt` with pinned versions once you're happy
- Push the LangGraph graph image using `app.get_graph().draw_mermaid()`
