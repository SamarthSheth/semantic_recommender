"""
Recommendation API.

Minimal backend that loads the precomputed FAISS index and serves
nearest-neighbor retrieval over natural language queries.
"""

import json
import os
from pathlib import Path
from contextlib import asynccontextmanager

import faiss
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ── Global state (loaded at startup) ─────────────────────────────────
model: SentenceTransformer = None
index: faiss.Index = None
movie_ids: list[int] = []
movies_lookup: dict[int, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, index, movie_ids, movies_lookup

    print("Loading sentence-transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading FAISS index...")
    index = faiss.read_index(str(EMBEDDINGS_DIR / "faiss_index.bin"))
    if hasattr(index, "nprobe"):
        index.nprobe = 10

    with open(EMBEDDINGS_DIR / "movie_ids.json") as f:
        movie_ids = json.load(f)

    print("Loading movie metadata...")
    df = pd.read_parquet(PROCESSED_DIR / "movies_final.parquet")
    for _, row in df.iterrows():
        movies_lookup[row["tmdbId"]] = {
            "tmdb_id": int(row["tmdbId"]),
            "title": row.get("title", row.get("title_tmdb", "")),
            "overview": row.get("overview", ""),
            "genres": json.loads(row["genres_tmdb"]) if isinstance(row["genres_tmdb"], str) else row["genres_tmdb"],
            "director": row.get("director", ""),
            "cast": json.loads(row["cast"]) if isinstance(row["cast"], str) else row["cast"],
            "avg_rating": round(float(row.get("avg_rating", 0)), 2),
            "num_ratings": int(row.get("num_ratings", 0)),
            "release_date": row.get("release_date", ""),
        }

    print(f"Ready: {index.ntotal} movies indexed")
    yield


# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Semantic Movie Recommender",
    description="Natural language movie recommendations via semantic search",
    lifespan=lifespan,
)

# CORS — env var for production, localhost fallback for dev
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5173"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=50)


class MovieResult(BaseModel):
    tmdb_id: int
    title: str
    overview: str
    genres: list[str]
    director: str | None
    cast: list[str]
    avg_rating: float
    num_ratings: int
    release_date: str
    similarity_score: float


class RecommendResponse(BaseModel):
    query: str
    results: list[MovieResult]


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    if index is None:
        return {"status": "loading"}
    return {"status": "ok", "movies_indexed": index.ntotal}


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    query_vec = model.encode(
        [req.query],
        normalize_embeddings=True,
    ).astype(np.float32)

    scores, indices = index.search(query_vec, req.top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        tmdb_id = movie_ids[idx]
        info = movies_lookup.get(tmdb_id, {})
        if info:
            results.append(MovieResult(
                similarity_score=round(float(score), 4),
                **info,
            ))

    return RecommendResponse(query=req.query, results=results)
