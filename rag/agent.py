from llm.groq import generate
from prompts.system_prompt import get_prompt
from rag.retriver import get_context


def run_agent(query: str) -> str:
    """Runs the RAG pipeline for a given query. Returns the answer text
    directly (unlike the original, which returned a langchain AIMessage
    with a .content attribute) - app.py's /query endpoint is updated
    to match."""
    context = get_context(query)
    prompt = get_prompt(context, query)
    return generate(prompt)