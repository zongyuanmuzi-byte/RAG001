import json
from core.embedding import get_embeddings
from core.faiss_index import create_index, add_vectors, save_index
from core.config import RAW_DOCS_PATH, DOCS_PATH, INDEX_PATH


def clean_text(text: str):
    text = text.strip()
    text = text.replace("\n", " ")
    while "  " in text:
        text = text.replace("  ", " ")
    return text


def chunk_text(text: str, chunk_size: int = 100, overlap: int = 20):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


with open(RAW_DOCS_PATH, "r", encoding="utf-8") as f:
    raw_docs = json.load(f)

all_chunks = []

for doc in raw_docs:
    cleaned = clean_text(doc)
    chunks = chunk_text(cleaned, chunk_size=100, overlap=20)
    all_chunks.extend(chunks)

vectors = get_embeddings(all_chunks)

index = create_index()
add_vectors(index, vectors)
save_index(index, INDEX_PATH)

with open(DOCS_PATH, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

print(f"建库完成，共生成 {len(all_chunks)} 个 chunks")