from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import faiss
import pickle
import numpy as np
import os

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

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
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

    return client.text_generation(
        prompt,
        max_new_tokens=500,
        temperature=0.3,
        do_sample=False
    ).strip()

@app.get("/ask")
async def ask(request: Request):
    q = request.query_params.get("q")
    if not q:
        return {"error": "Missing ?q= parameter"}
    return {"answer": ask_question(q)}