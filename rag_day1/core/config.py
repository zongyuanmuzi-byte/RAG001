import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ZHIPU_API_KEY")
BASE_URL = os.getenv("ZHIPU_BASE_URL")
EMBEDDING_MODEL = os.getenv("ZHIPU_EMBEDDING_MODEL")
CHAT_MODEL = os.getenv("ZHIPU_CHAT_MODEL", "glm-4-flash")

EMBEDDING_DIM = 2048

TOP_K = int(os.getenv("TOP_K", 5))
FINAL_TOP_N = int(os.getenv("FINAL_TOP_N", 3))

RAW_DOCS_PATH = os.getenv("RAW_DOCS_PATH", "data/raw_docs.json")
DOCS_PATH = os.getenv("DOCS_PATH", "data/docs.json")
INDEX_PATH = os.getenv("INDEX_PATH", "data/index.faiss")