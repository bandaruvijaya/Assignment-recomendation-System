import os
import pickle
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ---------- Config ----------
FAISS_INDEX_PATH = "embeddings/faiss.index"
META_PATH = "embeddings/metadata.pkl"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ---------- App ----------
app = FastAPI(title="SHL Assessment Recommender")

# ---------- Load resources ----------
index = faiss.read_index(FAISS_INDEX_PATH)
with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

embedder = SentenceTransformer(EMBED_MODEL_NAME)

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")

genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel("gemini-1.5-flash")

# ---------- Schemas ----------
class RecommendRequest(BaseModel):
    query: str

class Recommendation(BaseModel):
    assessment_name: str
    assessment_url: str

class RecommendResponse(BaseModel):
    recommendations: list[Recommendation]

# ---------- Helpers ----------
def fetch_text_from_url(url: str) -> str:
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.text[:12000]  # keep it bounded

def analyze_query_with_gemini(text: str) -> dict:
    prompt = f"""
You must return ONLY valid JSON.
No explanation. No markdown. No text outside JSON.

JSON format:
{{
  "technical_skills": [],
  "soft_skills": [],
  "focus": "K | P | A | MIX"
}}

Text:
{text}
"""
    try:
        resp = llm.generate_content(prompt)
        raw = resp.text.strip()

        # Extract JSON if Gemini adds noise
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON found")

        json_text = raw[start:end + 1]

        import json
        return json.loads(json_text)

    except Exception as e:
        print("⚠️ Gemini parsing failed:", e)
        return {
            "technical_skills": [],
            "soft_skills": [],
            "focus": "MIX"
        }


def search_faiss(query_text: str, top_k: int = 20):
    q_emb = embedder.encode([query_text], normalize_embeddings=True)
    q_emb = np.array(q_emb).astype("float32")
    scores, idxs = index.search(q_emb, top_k)
    results = []
    for i in idxs[0]:
        if i < 0:
            continue
        results.append(metadata[i])
    return results

def rerank(results, focus, min_k=5, max_k=10):
    # Simple, deterministic re-rank:
    # If soft skills present → ensure P included
    if focus == "P":
        primary = [r for r in results if r.get("test_type") == "P"]
        secondary = [r for r in results if r.get("test_type") != "P"]
    elif focus == "K":
        primary = [r for r in results if r.get("test_type") == "K"]
        secondary = [r for r in results if r.get("test_type") != "K"]
    else:  # MIX
        primary = results
        secondary = []

    ordered = primary + secondary
    return ordered[:max_k] if len(ordered) >= min_k else ordered

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    text = req.query.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty query")

    # URL handling
    if text.startswith("http://") or text.startswith("https://"):
        try:
            text = fetch_text_from_url(text)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")

    analysis = analyze_query_with_gemini(text)
    focus = analysis.get("focus", "MIX")

    candidates = search_faiss(text, top_k=20)
    ranked = rerank(candidates, focus)

    recs = [
        Recommendation(
            assessment_name=r["assessment_name"],
            assessment_url=r["url"]
        )
        for r in ranked
    ]

    return {"recommendations": recs}
