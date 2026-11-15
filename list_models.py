from groq import Groq
from src.config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)
for m in sorted(client.models.list().data, key=lambda x: x.id):
    print(m.id)