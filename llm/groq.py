from groq import Groq
from config import GROQ_API_KEY

MODEL_NAME = "llama-3.1-8b-instant"

_client = Groq(api_key=GROQ_API_KEY)


def generate(prompt: str) -> str:
    """Raw Groq chat completion call - replaces langchain_groq.ChatGroq
    (kept temperature=1, max_tokens=1024 to match the original config)."""
    response = _client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
        max_tokens=1024,
    )
    return response.choices[0].message.content