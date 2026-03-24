import faiss
import json
from core.embedding import get_embedding
from core.faiss_index import search
from core.config import TOP_K, FINAL_TOP_N

index = faiss.read_index("data/index.faiss")

with open("data/docs.json", "r", encoding="utf-8") as f:
    docs = json.load(f)


def simple_rerank(query: str, candidates: list[str]):
    query_terms = query.lower().split()

    scored = []
    for doc in candidates:
        text_lower = doc.lower()
        hit_count = sum(1 for term in query_terms if term in text_lower)
        scored.append((doc, hit_count))

    scored.sort(key=lambda x: x[1], reverse=True)
    reranked_docs = [item[0] for item in scored]

    return reranked_docs


def retrieve(query: str, k: int = TOP_K):
    query_vector = get_embedding(query)
    D, I = search(index, query_vector, k)

    candidates = []
    for idx in I[0]:
        if idx != -1:
            candidates.append(docs[idx])

    reranked_results = simple_rerank(query, candidates)

    return reranked_results[:FINAL_TOP_N]