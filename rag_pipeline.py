import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import openai

class RAGPipeline:
    def __init__(self, json_path, openai_api_key=None):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.IndexFlatL2(384)
        self.documents = []
        self.embeddings = []
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key
        self.load_and_embed(json_path)

    def load_and_embed(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for blog in data:
            content = blog.get("content", "")
            if content:
                self.documents.append(content)
                emb = self.model.encode(content)
                self.embeddings.append(emb)

        self.embeddings = np.vstack(self.embeddings)
        self.index.add(self.embeddings)

    def retrieve(self, query, top_k=3):
        query_vec = self.model.encode(query)
        D, I = self.index.search(np.array([query_vec]), top_k)
        return [self.documents[i] for i in I[0]]

    def generate_answer(self, query):
        context_docs = self.retrieve(query)
        context = "\n---\n".join(context_docs)
        prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}"

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content'].strip()
