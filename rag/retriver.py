from langchain_community.vectorstores import FAISS
from embeddings.embedding import embedding_model
from langchain_core.tools import create_retriever_tool


def get_vectorstore(chunks):
    """Create or load the vectorstore."""
    vectorstore = FAISS.from_texts(texts=chunks , embedding=embedding_model)
    return vectorstore


def load_vectorstore(index_path: str):
    """Load the existing vector store from faiss_index folder"""
    return FAISS.load_local(index_path, embeddings=embedding_model, allow_dangerous_deserialization=True)

def get_retriever():
    """Return the vector store as retriver """
    vectorstore = load_vectorstore("faiss_index")
    return create_retriever_tool(vectorstore.as_retriever(k=4),name="kb_serach" , description="search relevent context")



