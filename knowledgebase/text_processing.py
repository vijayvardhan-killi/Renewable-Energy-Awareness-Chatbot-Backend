from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os



def extract_text_from_pdf(pdf_path : str):
    """
    Extracts text from a PDF file.
    """
    with open(pdf_path , "rb") as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text


def extract_text_from_pdfs():
    """
    Extracts text from multiple PDF files and returns a list of texts.
    """
    folder = "knowledge_sources_pdfs"
    pdf_paths = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            pdf_paths.append(os.path.join(folder , file))

    
    texts = []
    for pdf_path in pdf_paths:
        print(f"Processing {pdf_path}...")
        text = extract_text_from_pdf(pdf_path)
        texts.append(text)
    return texts


def extract_chunks(texts):
    """
    Splits the text into chunks of approximately 1000 characters.
    """
    chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000 , chunk_overlap=200 , separators=["\n\n", "\n", ".", " ", ""])
    for text in texts:
        chunks.extend(text_splitter.split_text(text))
    return chunks

