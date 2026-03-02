"""
Compare baseline vs fine-tuned model on the same queries side by side.

Usage:
    python compare_queries.py

Run from the project root directory.
"""

import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).parent.parent.parent

# ── Load movie metadata for display ─────────────────────────────────
processed_dir = PROJECT_ROOT / "data" / "processed"
movies = pd.read_parquet(processed_dir / "movies_final.parquet")

id_to_info = {}
for _, row in movies.iterrows():
    genres = json.loads(row["genres_tmdb"]) if isinstance(row["genres_tmdb"], str) else row["genres_tmdb"]
    id_to_info[row["tmdbId"]] = {
        "title": row.get("title", row.get("title_tmdb", "Unknown")),
        "genres": genres,
    }

# ── Load baseline model + index ─────────────────────────────────────
print("Loading baseline model...")
baseline_model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings_dir = PROJECT_ROOT / "data" / "embeddings"
baseline_index = faiss.read_index(str(embeddings_dir / "faiss_index.bin"))
if hasattr(baseline_index, "nprobe"):
    baseline_index.nprobe = 10

with open(embeddings_dir / "movie_ids.json") as f:
    baseline_movie_ids = json.load(f)

# ── Load fine-tuned model + rebuild index ────────────────────────────
print("Loading fine-tuned model...")
finetuned_model = SentenceTransformer(str(PROJECT_ROOT / "models" / "baseline" / "model"))

# We need to rebuild the index with fine-tuned embeddings
print("Encoding movies with fine-tuned model...")
documents = movies["document"].tolist()
finetuned_movie_ids = movies["tmdbId"].tolist()

ft_embeddings = finetuned_model.encode(
    documents,
    batch_size=128,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True,
).astype(np.float32)

ft_index = faiss.IndexFlatIP(ft_embeddings.shape[1])
ft_index.add(ft_embeddings)
print(f"Fine-tuned index: {ft_index.ntotal} vectors\n")


# ── Run queries ──────────────────────────────────────────────────────

test_queries = [
    "a dark psychological thriller with an unreliable narrator",
    "feel-good comedy about friendship and growing up",
    "visually stunning sci-fi epic about space exploration",
    "quiet, melancholic drama set in a small town",
    "fast-paced heist movie with clever twists",
    # Additional vibes-based queries to test
    "something that feels like a rainy Sunday afternoon",
    "a movie that will make me ugly cry",
    "weird surreal film that messes with your head",
    "cozy nostalgic movie from the 80s or 90s",
    "intense courtroom drama with incredible acting",
]


def search(model, index, movie_ids, query, top_k=10):
    vec = model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            tmdb_id = movie_ids[idx]
            info = id_to_info.get(tmdb_id, {"title": "Unknown", "genres": []})
            results.append((score, info["title"], info["genres"]))
    return results


for query in test_queries:
    print("=" * 80)
    print(f'QUERY: "{query}"')
    print("=" * 80)

    baseline_results = search(baseline_model, baseline_index, baseline_movie_ids, query, top_k=10)
    finetuned_results = search(finetuned_model, ft_index, finetuned_movie_ids, query, top_k=10)

    # Print side by side
    print(f"\n  {'BASELINE':<42} {'FINE-TUNED':<42}")
    print(f"  {'-' * 40}   {'-' * 40}")

    for i in range(10):
        b_score, b_title, b_genres = baseline_results[i]
        f_score, f_title, f_genres = finetuned_results[i]

        b_str = f"[{b_score:.3f}] {b_title[:30]}"
        f_str = f"[{f_score:.3f}] {f_title[:30]}"

        print(f"  {i+1:>2}. {b_str:<40} {f_str:<40}")

    print()
