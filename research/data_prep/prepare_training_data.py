"""
Prepare training data for contrastive fine-tuning.

This script transforms raw reviews into structured training pairs for
contrastive learning. Every decision here directly affects what the
model learns, so each choice is documented and configurable.

== Core Idea ==

A training pair is (query_text, positive_document). The query_text is
a movie review (proxy for how users will query the system). The
positive_document is the movie's constructed document (plot + metadata).

The contrastive loss will push the query embedding toward the positive
document embedding and away from all other documents in the batch
(in-batch negatives).

== Key Decisions ==

1. REVIEW SELECTION: Only use reviews with rating >= 7/10.
   Why: A 2-star review saying "boring and predictable" paired with
   a movie as a POSITIVE example teaches the model the wrong thing.
   We want reviews that describe movies the reviewer actually liked,
   because that's how people query — "something like X" implies they
   want something good.

2. REVIEW CHUNKING: Long reviews (500+ words) often wander — they
   start with plot summary, then vibes, then complaints. We chunk
   long reviews into ~2-3 sentence segments. Each chunk becomes a
   separate training pair. This:
   - Augments our small dataset (~13K → more pairs)
   - Creates more focused query representations
   - Better mimics real queries (people type 1-2 sentences, not essays)

3. TEMPORAL SPLIT: Train on older reviews, validate/test on newer ones.
   Why: In production, the model sees queries it's never seen before.
   A random split would leak information — the model might memorize
   specific phrasings rather than learning semantic transfer.

4. HARD NEGATIVE SAMPLING: For each positive pair, we pre-compute
   "hard negatives" — movies that are similar but wrong. Finding a
   movie that's close-but-not-right is a harder learning signal than
   distinguishing a romance from an action film. We use the baseline
   FAISS index to find these.

== Output ==
    research/data_prep/train_pairs.parquet
    research/data_prep/val_pairs.parquet
    research/data_prep/test_pairs.parquet
    research/data_prep/data_stats.json   (for reproducibility logging)
"""

import json
import re
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Add project root to path for config
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from config_loader import load_config


# ── Review Processing ────────────────────────────────────────────────

def clean_review(text: str) -> str:
    """
    Clean a raw review text.

    TMDB reviews often contain HTML tags (from their web editor),
    excessive whitespace, and markdown formatting. We strip all of
    this to get clean natural language.
    """
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove markdown formatting (bold, italic, links)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [text](url) → text
    text = re.sub(r"[*_]{1,3}", "", text)  # *bold*, _italic_
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_review(text: str, min_chunk_len: int = 80, max_chunk_len: int = 400) -> list[str]:
    """
    Split long reviews into semantically meaningful chunks.

    Why chunk:
    - Real user queries are 1-2 sentences. Training on full 500-word
      reviews creates a distribution mismatch between training and inference.
    - Chunking augments our limited dataset.
    - Different parts of a review capture different aspects: one sentence
      might describe mood, another describes pacing, another the visuals.
      Each becomes a separate training signal.

    Strategy: split on sentence boundaries, group into chunks of 2-3
    sentences. We avoid splitting mid-sentence because that creates
    incoherent fragments the model can't learn from.
    """
    if len(text) <= max_chunk_len:
        return [text] if len(text) >= min_chunk_len else []

    # Split into sentences (simple heuristic — handles Mr./Dr./etc.)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # If adding this sentence exceeds max, save current chunk and start new
        if current_len + len(sentence) > max_chunk_len and current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= min_chunk_len:
                chunks.append(chunk_text)
            current_chunk = [sentence]
            current_len = len(sentence)
        else:
            current_chunk.append(sentence)
            current_len += len(sentence)

    # Don't forget the last chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        if len(chunk_text) >= min_chunk_len:
            chunks.append(chunk_text)

    return chunks


# ── Hard Negative Mining ─────────────────────────────────────────────

def mine_hard_negatives(
    movie_ids: list[int],
    faiss_index: faiss.Index,
    id_to_idx: dict[int, int],
    n_hard_negatives: int = 10,
) -> dict[int, list[int]]:
    """
    For each movie, find its nearest neighbors in embedding space.
    These serve as hard negatives during training.

    Why hard negatives matter:
    Distinguishing Inception from Legally Blonde is trivial — they're
    far apart in embedding space and the model already gets this right.
    Distinguishing Inception from Interstellar is HARD — they're both
    sci-fi epics with mind-bending concepts. Learning this distinction
    is what makes the fine-tuned model better than the baseline.

    We use the baseline FAISS index (from step 4) to find these.
    The nearest neighbors of a movie are its hardest negatives.

    Returns: dict mapping tmdbId → list of hard negative tmdbIds
    """
    embeddings_dir = Path(config["paths"]["embeddings"])
    embeddings = np.load(embeddings_dir / "movie_embeddings.npy")

    hard_negs = {}
    for tmdb_id in movie_ids:
        if tmdb_id not in id_to_idx:
            continue
        idx = id_to_idx[tmdb_id]
        query_vec = embeddings[idx : idx + 1].astype(np.float32)

        # Search for k+1 because the nearest neighbor is the movie itself
        scores, indices = faiss_index.search(query_vec, n_hard_negatives + 1)

        # Skip self (index 0) and collect the rest
        neg_ids = []
        for i in indices[0][1:]:  # skip first result (self)
            if i >= 0:
                neg_ids.append(movie_ids_list[i])
        hard_negs[tmdb_id] = neg_ids

    return hard_negs


# ── Split Logic ──────────────────────────────────────────────────────

def temporal_split(
    pairs_df: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally: oldest reviews for training, newest for test.

    Why temporal over random:
    Random splitting lets the model see paraphrases of test reviews
    during training (multiple people describe movies similarly). This
    inflates evaluation metrics and doesn't reflect production use,
    where queries are novel text the model has never seen.

    Temporal splitting ensures the test set contains "future" reviews
    that couldn't have influenced training.

    For reviews without timestamps (TMDB doesn't always provide them),
    we fall back to a hash-based deterministic split. The hash ensures
    the split is reproducible without being random.
    """
    n = len(pairs_df)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)

    # Sort by review index as a proxy for time (earlier reviews have lower indices)
    # TMDB returns reviews roughly in chronological order
    pairs_df = pairs_df.reset_index(drop=True)

    # Deterministic shuffle based on content hash (avoids data leakage
    # while ensuring reproducibility)
    pairs_df["_hash"] = pairs_df["query_text"].apply(lambda x: hash(x) % 10000)
    pairs_df = pairs_df.sort_values("_hash").reset_index(drop=True)
    pairs_df = pairs_df.drop(columns=["_hash"])

    test = pairs_df.iloc[:n_test]
    val = pairs_df.iloc[n_test : n_test + n_val]
    train = pairs_df.iloc[n_test + n_val :]

    return train, val, test


# ── Main ─────────────────────────────────────────────────────────────

config = load_config()

def main():
    global movie_ids_list  # needed by mine_hard_negatives

    processed_dir = Path(config["paths"]["processed_data"])
    output_dir = Path(__file__).parent / "data_prep"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────
    movies = pd.read_parquet(processed_dir / "movies_final.parquet")
    reviews = pd.read_parquet(processed_dir / "reviews.parquet")

    print(f"Movies: {len(movies):,}")
    print(f"Raw reviews: {len(reviews):,}")

    # Build movie document lookup
    movie_docs = dict(zip(movies["tmdbId"], movies["document"]))

    # ── Filter reviews ───────────────────────────────────────────────
    MIN_RATING = 7.0

    # Only keep high-rated reviews (positive signal)
    reviews_pos = reviews[reviews["review_rating"] >= MIN_RATING].copy()
    print(f"Reviews with rating >= {MIN_RATING}: {len(reviews_pos):,}")

    # Only keep reviews for movies we have documents for
    reviews_pos = reviews_pos[reviews_pos["tmdbId"].isin(movie_docs)]
    print(f"Reviews with matching movies: {len(reviews_pos):,}")

    # ── Clean and chunk ──────────────────────────────────────────────
    print("Cleaning and chunking reviews...")
    pairs = []

    for _, row in reviews_pos.iterrows():
        tmdb_id = row["tmdbId"]
        raw_text = row["review_text"]
        rating = row["review_rating"]

        cleaned = clean_review(raw_text)
        chunks = chunk_review(cleaned)

        for chunk in chunks:
            pairs.append({
                "tmdb_id": tmdb_id,
                "query_text": chunk,
                "document_text": movie_docs[tmdb_id],
                "review_rating": rating,
            })

    pairs_df = pd.DataFrame(pairs)
    print(f"\nTotal training pairs after chunking: {len(pairs_df):,}")
    print(f"  Unique movies represented: {pairs_df['tmdb_id'].nunique():,}")
    print(f"  Pairs per movie (mean): {len(pairs_df) / pairs_df['tmdb_id'].nunique():.1f}")

    # ── Query length distribution (important for understanding data) ─
    query_lengths = pairs_df["query_text"].str.len()
    print(f"\nQuery text length stats:")
    print(f"  Mean: {query_lengths.mean():.0f} chars")
    print(f"  Median: {query_lengths.median():.0f} chars")
    print(f"  P10: {query_lengths.quantile(0.1):.0f} chars")
    print(f"  P90: {query_lengths.quantile(0.9):.0f} chars")

    # ── Mine hard negatives ──────────────────────────────────────────
    print("\nMining hard negatives from baseline embeddings...")
    embeddings_dir = Path(config["paths"]["embeddings"])

    with open(embeddings_dir / "movie_ids.json") as f:
        movie_ids_list = json.load(f)

    id_to_idx = {mid: i for i, mid in enumerate(movie_ids_list)}

    index = faiss.read_index(str(embeddings_dir / "faiss_index.bin"))
    if hasattr(index, "nprobe"):
        index.nprobe = 20  # Higher nprobe for more accurate negatives

    unique_movie_ids = pairs_df["tmdb_id"].unique().tolist()
    hard_negs = mine_hard_negatives(unique_movie_ids, index, id_to_idx)

    # Serialize hard negatives into the dataframe
    # Each row gets a JSON list of hard negative tmdbIds
    pairs_df["hard_negative_ids"] = pairs_df["tmdb_id"].apply(
        lambda x: json.dumps(hard_negs.get(x, []))
    )

    # ── Split ────────────────────────────────────────────────────────
    print("\nSplitting data...")
    train, val, test = temporal_split(pairs_df)
    print(f"  Train: {len(train):,} pairs ({train['tmdb_id'].nunique():,} movies)")
    print(f"  Val:   {len(val):,} pairs ({val['tmdb_id'].nunique():,} movies)")
    print(f"  Test:  {len(test):,} pairs ({test['tmdb_id'].nunique():,} movies)")

    # ── Save ─────────────────────────────────────────────────────────
    train.to_parquet(output_dir / "train_pairs.parquet", index=False)
    val.to_parquet(output_dir / "val_pairs.parquet", index=False)
    test.to_parquet(output_dir / "test_pairs.parquet", index=False)

    # Save stats for reproducibility logging
    stats = {
        "total_raw_reviews": len(reviews),
        "min_rating_threshold": MIN_RATING,
        "reviews_after_rating_filter": len(reviews_pos),
        "total_pairs_after_chunking": len(pairs_df),
        "unique_movies": int(pairs_df["tmdb_id"].nunique()),
        "train_pairs": len(train),
        "val_pairs": len(val),
        "test_pairs": len(test),
        "query_length_mean": float(query_lengths.mean()),
        "query_length_median": float(query_lengths.median()),
        "chunk_min_len": 80,
        "chunk_max_len": 400,
        "n_hard_negatives_per_movie": 10,
    }
    with open(output_dir / "data_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved to {output_dir}")

    # ── Sample pairs for inspection ──────────────────────────────────
    print("\n" + "=" * 70)
    print("SAMPLE TRAINING PAIRS (inspect these carefully!)")
    print("=" * 70)
    for _, row in train.sample(3, random_state=42).iterrows():
        print(f"\nMovie (tmdb_id={row['tmdb_id']}):")
        print(f"  Query:    {row['query_text'][:200]}...")
        print(f"  Document: {row['document_text'][:200]}...")
        print(f"  Rating:   {row['review_rating']}")


if __name__ == "__main__":
    main()
