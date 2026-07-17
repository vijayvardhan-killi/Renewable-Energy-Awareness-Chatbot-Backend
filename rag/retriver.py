from embeddings.embedding import embed_query
from rag.vectorstore import get_index

TOP_K = 4
TEXT_KEY = "text"  # matches text_key="text" used when the index was built


def get_context(question: str) -> str:
    """Embed the question, query Pinecone directly, and join matched
    chunk text - same effective behavior as the original
    vectorstore.as_retriever(k=4) + create_retriever_tool, without the
    langchain_core.tools / langchain_pinecone dependency chain."""
    vector = embed_query(question)
    results = get_index().query(
        vector=vector,
        top_k=TOP_K,
        include_metadata=True,
    )
    chunks = [
        match["metadata"].get(TEXT_KEY, "")
        for match in results.get("matches", [])
        if match.get("metadata")
    ]
    return "\n\n".join(chunks)