from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
import os

load_dotenv()
 
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
    )

def get_vectorstore(chunks):
    """Create or load the vectorstore."""
    vectorstore = FAISS.from_texts(texts=chunks , embedding=embedding_model)
    
    return vectorstore


def load_vectorstore(index_path: str):
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.load_local(index_path, embeddings=embedding_model, allow_dangerous_deserialization=True)



