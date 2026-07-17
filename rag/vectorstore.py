from pinecone import Pinecone
from config import PINECONE_API_KEY

INDEX_NAME = "renewable-energy-kb"

_pc = Pinecone(api_key=PINECONE_API_KEY)
_index = _pc.Index(INDEX_NAME)


def get_index():
    """Raw Pinecone index handle - no langchain_pinecone wrapper
    (which pulls in langchain-openai -> openai + tiktoken, and numpy)."""
    return _index