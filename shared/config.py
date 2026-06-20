"""
Shared settings used by ALL THREE framework versions (LangChain, LangGraph, CrewAI).

Keeping the model, the input feature, and the prompts in one place means the
three implementations are doing exactly the same job. Only the orchestration
code (how Agent 1 talks to Agent 2) is different between folders.
"""

# --- Model settings -----------------------------------------------------
MODEL_NAME = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434"

# --- The input every agent pipeline works on -----------------------------
SAMPLE_FEATURE = """
Feature: User Login
As a registered user, I want to log in with my email and password
so that I can access my account dashboard.

Acceptance Criteria:
- User can log in with a valid email and password
- User sees an error message for invalid credentials
- Account locks after 5 failed login attempts
- Password field must be masked
"""

# --- Agent 1: Requirements Analyst ---------------------------------------
ANALYST_PROMPT = """You are a QA Requirements Analyst.

Read the feature description below and extract a short, numbered list of
TEST SCENARIOS only (no detailed steps yet). Cover positive, negative, and
edge cases.

Feature description:
{feature}
"""

# --- Agent 2: Test Case Writer --------------------------------------------
WRITER_PROMPT = """You are a QA Test Case Writer.

You have been handed a list of test scenarios from the Requirements Analyst.
For EACH scenario, write a detailed test case using exactly this format:

Test ID: TC-XX
Title: <short title>
Steps:
 1. ...
 2. ...
Expected Result: ...

Scenarios from the Analyst:
{scenarios}
"""
