from flask import Flask , jsonify , request
from knowledgebase.text_processing import extract_text_from_pdfs , extract_chunks
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os

app = Flask(__name__)


# ========== Global Variables ==========
vectorstore = None
qa_chain = None

# === Initialize vectorstore once on server startup ===
def initialize_vectorstore():
    global vectorstore , qa_chain

    #load from saved index if avilable
    if os.path.exists("faiss_index"):
        print("Loading existing FAISS index...")
        from knowledgebase.vectorstore import load_vectorstore
        vectorstore = load_vectorstore("faiss_index")
    else:
        print("Creating FAISS index from PDFs...")
        texts = extract_text_from_pdfs()
        print(f"{len(texts)} PDFs processed.")
        chunks = extract_chunks(texts)
        from knowledgebase.vectorstore import get_vectorstore
        vectorstore = get_vectorstore(chunks)
        vectorstore.save_local("faiss_index")

def create_qa_chain():
    global qa_chain
    
    prompt_template = """
You are GreenGuide, a helpful and knowledgeable assistant that explains renewable energy topics clearly and accurately.

You must answer based **only** on the provided documents. If the answer is not found in the documents, respond with: "I'm not sure about that. Would you like to explore something else related to renewable energy?"

Always explain things simply and with practical examples. Your audience may include school students, college youth, or the general public who are curious about solar, wind, hydro, or other renewable resources.

Only include information that is present in the retrieved context below.

Context:
{context}

Question:
{question}

Answer:
"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-001"
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

initialize_vectorstore()
create_qa_chain()



# === Home endpoint ===
@app.route('/' , methods = ['GET'])
def index():
    
    return jsonify({'message': 'Welcome to the CHATBOT API!'})


# === Query endpoint ===
@app.route("/query", methods=["POST"])
def query():
    user_input = request.json.get("question", "")
    if not user_input:
        return jsonify({"error": "Question is required"}), 400

    result = qa_chain.invoke({"query": user_input})
    return jsonify({
        "answer": result["result"],
        # "sources": [doc.page_content[:300] for doc in result["source_documents"]] #for debug purpose only uncomment
    })



if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000 ,debug=True)