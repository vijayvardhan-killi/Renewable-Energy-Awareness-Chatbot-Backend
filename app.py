from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from schemas.query import QueryRequest
from rag.agent import run_agent

app = FastAPI(title="Renewable Energy Awareness Chatbot API (lite)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    return {"message": "Welcome to CHATBOT API (lite build)"}


@app.post("/query")
def query(payload: QueryRequest):
    if not payload.question:
        return {"error": "Question required"}
    answer = run_agent(payload.question)
    return {"answer": answer}