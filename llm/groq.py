from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

groq_model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=1,
    max_tokens=1024,
)
 