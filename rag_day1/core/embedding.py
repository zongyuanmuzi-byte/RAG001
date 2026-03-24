from openai import OpenAI
from core.config import API_KEY,EMBEDDING_MODEL,BASE_URL

client = OpenAI(
    api_key=API_KEY,
    base_url= BASE_URL
    )

def get_embedding(text:str):

    response= client.embeddings.create (
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding
