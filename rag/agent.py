from llm.groq import groq_model

from prompts.system_prompt import get_prompt
from rag.retriver import get_retriever

def search(query:str):
    """Gets the relvent information from already acquired Knowledgebase"""
    retriever = get_retriever() 
    docs = retriever.invoke(query)
    return "\n\n".join(d for d in docs)

def run_agent(query ) -> str:
    """runs the agent for given query"""
    model= groq_model
    context = search(query)
    prompt = get_prompt(context , query)
    response = model.invoke(prompt)
    return response
