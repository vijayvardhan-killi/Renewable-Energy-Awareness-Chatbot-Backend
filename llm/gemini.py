from langchain_google_genai import ChatGoogleGenerativeAI
from config import GOOGLE_API_KEY


gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    google_api_key=GOOGLE_API_KEY,
    )