import faiss
import numpy as np


def create_index(dimension: int):
    index = faiss.IndexFlatL2(dimension)
    return index


def add_vectors(index, vectors):
    vectors = np.array(vectors).astype("float32")
    index.add(vectors)


def search(index, query_vector, k=3):
    query_vector = np.array([query_vector]).astype("float32")
    D, I = index.search(query_vector, k)
    return D, I