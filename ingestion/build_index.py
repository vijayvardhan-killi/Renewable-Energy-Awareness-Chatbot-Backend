from ingestion.text_processing import extract_text_from_pdfs, extract_chunks
from rag.vectorstore import get_vectorstore


def run_ingestion():
    # Extract text from PDF's
    texts = extract_text_from_pdfs("knowledge_sources_pdfs")

    # Break the text into Chunks
    chunks = extract_chunks(texts)

    # Create vectorstore
    vectorstore = get_vectorstore()
    vectorstore.add_texts(chunks)

    print("Ingestion complete")

if __name__ == "__main__":
    run_ingestion()


