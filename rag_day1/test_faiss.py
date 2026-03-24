import faiss
import numpy as np

vectors=[
    [1.0,1.0],
    [0.0,1.0],
    [0.9,0.1]
]
vectors =np.array(vectors).astype("float32")

index = faiss.IndexFlatL2(2)

index.add(vectors)

query =np.array([[1.0,0.0]]).astype("float32")

D, I = index.search(query, k=2)

print("距离:", D)
print("索引:", I)