"""
Makes sure the project root is on sys.path so imports like
`from shared.config import ...` and `from langgraph_version.main import ...`
work when pytest runs, regardless of which folder you run it from.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
