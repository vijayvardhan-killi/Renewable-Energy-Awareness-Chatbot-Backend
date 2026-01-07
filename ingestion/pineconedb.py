import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "renewable-energy-kb"

def create_index(index_name: str):
    if pc.has_index(index_name):
        print("Index already exists")
        return

    pc.create_index(
        name=index_name,
        dimension=384,   # HF MiniLM
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
    )
    print("Index created")

create_index(index_name=index_name)
