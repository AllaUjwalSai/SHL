import faiss
import numpy as np

class Retriever:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.documents = []

    def build_index(self, embeddings, documents):
        self.index.add(np.array(embeddings).astype("float32"))
        self.documents = documents

    def query(self, query_embedding, k=5):
        D, I = self.index.search(np.array([query_embedding]).astype("float32"), k)
        return [self.documents[i] for i in I[0]]
