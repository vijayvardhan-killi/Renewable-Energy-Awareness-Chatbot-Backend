from huggingface_hub import InferenceClient
from config import HF_API_KEY

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_client = InferenceClient(provider="hf-inference", api_key=HF_API_KEY)


def embed_query(text: str) -> list[float]:
    """
    Mirrors langchain_huggingface.HuggingFaceEndpointEmbeddings.embed_query()
    exactly: same client call, same newline replacement, same model —
    so vectors are identical to what's already indexed in Pinecone.
    """
    text = text.replace("\n", " ")
    response = _client.feature_extraction(text=text, model=MODEL_NAME)
    return response.tolist() if hasattr(response, "tolist") else list(response)