from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from embeddings.embedding import embedding_model
from ingestion.pineconedb import index_name
import os

def get_vectorstore():
    """Create the vectorstore using PINECONE cloud."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(index_name)

    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embedding_model,
        text_key="text",
    )
    
    return vectorstore