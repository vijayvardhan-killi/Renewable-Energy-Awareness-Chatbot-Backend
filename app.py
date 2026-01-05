from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from schemas.query import QueryRequest
from rag.agent import run_agent

app = FastAPI(title="Renewable Energy Awareness Chatbot API")

# CORS 
# Enable CORS for all routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)





# === Home endpoint ===
@app.get("/")
def health_check():
    return {"message": "Welocome to CHATBOT API"}


# === Query endpoint ===
@app.post("/query")
def query(payload : QueryRequest):
    if not payload.question:
        return {"error" : "Question required"}
    result = run_agent(payload.question)

    return {"answer" : result.content}