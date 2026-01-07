from langchain_community.vectorstores import FAISS
from embeddings.embedding import embedding_model
from langchain_core.tools import create_retriever_tool
from rag.vectorstore import get_vectorstore

from pinecone import Pinecone
import os

# Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "renewable-energy-kb"


def load_vectorstore(index_path: str):
    """Load the existing vector store from faiss_index folder"""

    return FAISS.load_local(index_path, embeddings=embedding_model, allow_dangerous_deserialization=True)


def get_retriever():
    """Return the vector store as retriver """
    vectorstore = get_vectorstore()
    return create_retriever_tool(
        vectorstore.as_retriever(k=4),
        name="kb_serach", 
        description="Search renewable energy knowledge base",
    )



