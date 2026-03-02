"""
Step 4: Compute embeddings and build FAISS index.

What this does:
    1. Loads the movie documents from step 3
    2. Encodes each document using a pretrained sentence-transformer
    3. Builds a FAISS index for fast nearest-neighbor retrieval
    4. Saves both the embeddings and the index to disk

Why a frozen pretrained model for the baseline?
    Sentence-transformers (trained on semantic similarity tasks) already
    understand that "a melancholic thriller" is closer to "a dark,
    brooding crime film" than to "a lighthearted romantic comedy".

    By NOT fine-tuning, we establish what's achievable for free. This
    baseline has two possible outcomes, both good:
    1. It works well → "pretrained representations transfer to this domain"
       is a valid finding. We analyze WHY and WHERE it succeeds/fails.
    2. It works poorly → clear motivation for contrastive fine-tuning,
       and we can measure exactly how much fine-tuning helps.

    Jumping straight to fine-tuning without a baseline is bad science.

Why FAISS over brute-force cosine similarity?
    With ~10K movies and 384-dim embeddings, brute force is actually fast
    enough (~5ms). We use FAISS anyway because:
    1. It scales — if you later expand to 100K+ movies, brute force dies
    2. It shows you understand retrieval engineering, not just ML
    3. The IVF index teaches you about the accuracy/speed tradeoff

Output:
    - data/embeddings/movie_embeddings.npy — raw embedding matrix
    - data/embeddings/faiss_index.bin — FAISS index file
    - data/embeddings/movie_ids.json — tmdbId ordering (row i in the
      matrix corresponds to movie_ids[i])
"""

import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config_loader import load_config


def compute_embeddings(
    documents: list[str],
    model_name: str,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Encode documents using a sentence-transformer.

    Decision: batch_size=64 balances GPU memory (if available) with
    throughput. sentence-transformers handles device placement automatically
    (GPU if available, else CPU).

    Decision: normalize embeddings to unit length so that cosine similarity
    equals dot product. This is important because FAISS's IndexFlatIP
    (inner product) is faster than IndexFlatL2 + cosine conversion.
    """
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Encoding {len(documents):,} documents...")
    # show_progress_bar uses tqdm internally
    embeddings = model.encode(
        documents,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2 normalize → cosine sim = dot product
        convert_to_numpy=True,
    )

    print(f"Embedding shape: {embeddings.shape}")
    return embeddings


def build_faiss_index(
    embeddings: np.ndarray,
    index_type: str,
    nlist: int,
) -> faiss.Index:
    """
    Build a FAISS index for fast retrieval.

    Decision: IVFFlat index. How it works:
    1. K-means clusters the embedding space into `nlist` regions (Voronoi cells)
    2. Each movie is assigned to the nearest cluster
    3. At query time, we only search `nprobe` clusters (not all of them)

    This is an approximation — we might miss a relevant movie if it's in
    a cluster we didn't search. But with nlist=100 and nprobe=10, recall
    is typically >95% while being much faster than exhaustive search.

    For our catalog size (~10K), this is honestly overkill. But it
    demonstrates you understand the engineering tradeoffs in retrieval
    systems, which matters for the quant firm audience.
    """
    dim = embeddings.shape[1]
    embeddings = embeddings.astype(np.float32)  # FAISS requires float32

    if index_type == "Flat":
        # Exact search — use for small catalogs or as a correctness check
        index = faiss.IndexFlatIP(dim)  # IP = inner product (= cosine for normalized vecs)
        index.add(embeddings)
    elif index_type == "IVFFlat":
        # Approximate search with inverted file index
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        # IVF requires training (learning the cluster centroids)
        print(f"Training IVF index with {nlist} clusters...")
        index.train(embeddings)
        index.add(embeddings)
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    print(f"Index built: {index.ntotal} vectors, type={index_type}")
    return index


def test_retrieval(
    index: faiss.Index,
    model_name: str,
    movie_ids: list[int],
    movies_df: pd.DataFrame,
    nprobe: int,
):
    """
    Quick sanity test: run a few natural language queries and show results.

    This is NOT formal evaluation — it's a smoke test to make sure the
    pipeline isn't broken before we invest time in systematic benchmarking.
    """
    model = SentenceTransformer(model_name)

    if hasattr(index, "nprobe"):
        index.nprobe = nprobe

    # Map tmdbId to movie info for display
    id_to_info = {}
    for _, row in movies_df.iterrows():
        id_to_info[row["tmdbId"]] = {
            "title": row.get("title", row.get("title_tmdb", "Unknown")),
            "genres_tmdb": row["genres_tmdb"],
        }

    test_queries = [
        "a dark psychological thriller with an unreliable narrator",
        "feel-good comedy about friendship and growing up",
        "visually stunning sci-fi epic about space exploration",
        "quiet, melancholic drama set in a small town",
        "fast-paced heist movie with clever twists",
    ]

    print("\n" + "=" * 70)
    print("SANITY CHECK: Baseline retrieval results")
    print("=" * 70)

    for query in test_queries:
        # Encode query
        query_vec = model.encode([query], normalize_embeddings=True).astype(np.float32)

        # Search
        scores, indices = index.search(query_vec, k=5)

        print(f"\nQuery: \"{query}\"")
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            tmdb_id = movie_ids[idx]
            info = id_to_info.get(tmdb_id, {"title": "Unknown", "genres_tmdb": "[]"})
            print(f"  {rank+1}. [{score:.3f}] {info['title']} ({info['genres_tmdb']})")


def main():
    config = load_config()
    emb_dir = Path(config["paths"]["embeddings"])

    # Load processed movies
    movies_path = Path(config["paths"]["processed_data"]) / "movies_final.parquet"
    if not movies_path.exists():
        raise FileNotFoundError(
            f"{movies_path} not found. Run 03_merge_and_build_documents.py first."
        )
    df = pd.read_parquet(movies_path)
    print(f"Loaded {len(df):,} movies")

    documents = df["document"].tolist()
    movie_ids = df["tmdbId"].tolist()

    # Compute embeddings
    model_name = config["model"]["baseline_model"]
    embeddings = compute_embeddings(documents, model_name)

    # Save embeddings
    np.save(emb_dir / "movie_embeddings.npy", embeddings)
    with open(emb_dir / "movie_ids.json", "w") as f:
        json.dump(movie_ids, f)
    print(f"Saved embeddings to {emb_dir}")

    # Build FAISS index
    index = build_faiss_index(
        embeddings,
        index_type=config["model"]["faiss_index_type"],
        nlist=config["model"]["faiss_nlist"],
    )
    faiss.write_index(index, str(emb_dir / "faiss_index.bin"))
    print(f"Saved FAISS index to {emb_dir / 'faiss_index.bin'}")

    # Sanity test
    test_retrieval(
        index, model_name, movie_ids, df,
        nprobe=config["model"]["faiss_nprobe"],
    )


if __name__ == "__main__":
    main()
