from src.config import GROQ_API_KEY, GROQ_MODEL
from langchain_groq import ChatGroq

llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)
print(llm.invoke("Reply with exactly: Groq is working").content)