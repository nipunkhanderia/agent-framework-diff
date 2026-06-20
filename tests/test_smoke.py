"""
Smoke tests
===========
These are intentionally light. They do NOT call the real LLM (Ollama isn't
guaranteed to be running in CI), they just check that everything is wired up
correctly: prompts have the right placeholders, the LangGraph graph compiles,
and the CrewAI crew is built with the right agents/tasks.
"""
from shared.config import ANALYST_PROMPT, WRITER_PROMPT, SAMPLE_FEATURE


def test_sample_feature_is_not_empty():
    assert len(SAMPLE_FEATURE.strip()) > 0


def test_analyst_prompt_has_placeholder():
    assert "{feature}" in ANALYST_PROMPT


def test_writer_prompt_has_placeholder():
    assert "{scenarios}" in WRITER_PROMPT


def test_langgraph_graph_compiles():
    from langgraph_version.main import build_graph

    app = build_graph()
    assert app is not None


def test_crewai_crew_has_two_agents_and_two_tasks():
    from crewai_version.main import crew

    assert len(crew.agents) == 2
    assert len(crew.tasks) == 2
