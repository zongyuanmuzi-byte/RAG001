from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from services.rag_service import rag_answer
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def root():
    return {"message": "RAG system running"}

@app.post("/chat")
def chat(request: ChatRequest):
    query = request.message.strip()

    if not query:
        logging.warning("收到空 message 请求")
        raise HTTPException(status_code=400, detail="message cannot be empty")

    logging.info(f"收到用户问题: {query}")

    try:
        result = rag_answer(query)
        logging.info("RAG answer generated successfully")
        return result
    except Exception as e:
        logging.error(f"RAG error: {e}")
        raise HTTPException(status_code=500, detail="server error")