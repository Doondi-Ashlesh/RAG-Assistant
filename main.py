from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import faiss
import pickle
import numpy as np
import os
import uvicorn

app = FastAPI()

# Allow frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your domain for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.getenv("HF_TOKEN")

#embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

index = faiss.read_index("index.faiss")

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

client = InferenceClient(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    token=HF_TOKEN
)

def search_faiss(query, top_k=5):
    query_embed = embedding_model.encode([query])
    D, I = index.search(np.array(query_embed), top_k)
    return [chunks[i] for i in I[0]]

def ask_question(query):
    context = search_faiss(query)
    prompt = f"""Answer the question based on the context below.

Context:
{context}

Question: {query}
Answer:"""

    response = client.text_generation(
        prompt,
        max_new_tokens=500,
        temperature=0.3,
        do_sample=False
    )
    # client.text_generation returns a string
    return response.strip()

@app.get("/ask")
async def ask(request: Request):
    q = request.query_params.get("q")
    if not q:
        return {"error": "Missing ?q= parameter"}
    answer = ask_question(q)
    return {"answer": answer}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
