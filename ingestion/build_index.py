from ingestion.text_processing import extract_text_from_pdfs, extract_chunks
from rag.retriver import get_vectorstore
# Build the vectorstore (faiss_index) offline

texts = extract_text_from_pdfs("knowledge_sources_pdfs")
print(f"{len(texts)} PDFs processed.")
chunks = extract_chunks(texts)
vectorstore = get_vectorstore(chunks)
vectorstore.save_local("faiss_index")