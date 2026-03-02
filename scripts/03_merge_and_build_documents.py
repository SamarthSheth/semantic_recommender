"""
Step 3: Merge MovieLens + TMDB into final dataset and construct text documents.

This is the most important data decision in the pipeline:
**What text represents each movie in embedding space?**

We construct a "document" for each movie by combining:
    1. Plot overview (the core semantic content)
    2. Genre labels (categorical context)
    3. Keywords (thematic tags)
    4. Director + cast (people as signal)

Why combine them rather than embed separately?
    A single combined document captures the full "meaning" of a movie in
    one embedding vector. This keeps retrieval simple: one query vector
    vs one movie vector, cosine similarity, done.

    The alternative — separate embeddings per field, combined at retrieval
    time — is more flexible but adds complexity we don't need for the
    baseline. We can explore that in the research phase.

Why this specific format?
    We structure the document as natural language rather than just
    concatenating fields. "A sci-fi thriller directed by Denis Villeneuve
    starring Timothée Chalamet" gives the language model more context than
    "sci-fi, thriller, Denis Villeneuve, Timothée Chalamet". The model
    understands relationships between tokens better when they're in
    natural sentence structure.

Output: data/processed/movies_final.parquet
    Contains all metadata + a 'document' column ready for embedding.
"""

import json
from pathlib import Path

import pandas as pd

from config_loader import load_config


def construct_document(row: pd.Series) -> str:
    """
    Build a natural-language document from movie metadata.

    This is the text that gets embedded. Every design choice here
    affects what the model "knows" about a movie.

    Decision: lead with overview because it carries the most semantic
    weight. Follow with structured metadata in natural language.
    Don't include reviews — those live on the query side (they're
    what people *say about* movies, similar to how people will query).
    """
    parts = []

    # Overview is the primary content
    overview = row.get("overview", "")
    if overview and len(str(overview)) > 10:
        parts.append(str(overview))

    # Genres as natural language
    genres = json.loads(row["genres_tmdb"]) if isinstance(row["genres_tmdb"], str) else row["genres_tmdb"]
    if genres:
        parts.append(f"Genre: {', '.join(genres)}.")

    # Keywords capture themes the overview might miss
    keywords = json.loads(row["keywords"]) if isinstance(row["keywords"], str) else row["keywords"]
    if keywords:
        # Cap at 10 keywords to avoid noise
        parts.append(f"Themes: {', '.join(keywords[:10])}.")

    # Director and cast — people carry strong semantic signal
    director = row.get("director", "")
    if director and str(director) != "nan":
        parts.append(f"Directed by {director}.")

    cast = json.loads(row["cast"]) if isinstance(row["cast"], str) else row["cast"]
    if cast:
        parts.append(f"Starring {', '.join(cast[:3])}.")

    return " ".join(parts)


def main():
    config = load_config()
    processed_dir = Path(config["paths"]["processed_data"])

    # Load both sources
    movielens = pd.read_parquet(processed_dir / "movielens_filtered.parquet")
    tmdb = pd.read_parquet(processed_dir / "tmdb_metadata.parquet")

    print(f"MovieLens movies: {len(movielens):,}")
    print(f"TMDB movies: {len(tmdb):,}")

    # Merge on tmdbId
    # Decision: inner join — only keep movies present in BOTH sources.
    # A movie in MovieLens but not TMDB has no text to embed.
    # A movie in TMDB but not MovieLens has no rating signal.
    df = movielens.merge(tmdb, on="tmdbId", how="inner")
    print(f"After merge: {len(df):,} movies")

    # Filter: must have a real overview
    min_len = config["data"]["min_overview_length"]
    before = len(df)
    df = df[df["overview"].str.len() >= min_len]
    print(f"After overview filter (min {min_len} chars): {len(df):,} "
          f"(dropped {before - len(df):,})")

    # Construct the embedding document
    print("Constructing documents...")
    df["document"] = df.apply(construct_document, axis=1)

    # Quick sanity checks
    print(f"\nDocument length stats:")
    doc_lengths = df["document"].str.len()
    print(f"  Mean: {doc_lengths.mean():.0f} chars")
    print(f"  Median: {doc_lengths.median():.0f} chars")
    print(f"  Min: {doc_lengths.min()} chars")
    print(f"  Max: {doc_lengths.max()} chars")

    # Show a sample document
    sample = df.sample(1).iloc[0]
    print(f"\nSample movie: {sample['title']}")
    print(f"Document: {sample['document'][:500]}")

    # Extract review texts into a separate file for contrastive training later
    # Decision: separate file because reviews are only needed for fine-tuning,
    # not for the baseline. Keeping them separate means the main dataset stays
    # small and fast to load during experimentation.
    print("\nExtracting reviews...")
    review_records = []
    for _, row in df.iterrows():
        reviews = json.loads(row["reviews"]) if isinstance(row["reviews"], str) else row["reviews"]
        for review in reviews:
            if review.get("text") and review.get("rating") is not None:
                review_records.append({
                    "tmdbId": row["tmdbId"],
                    "review_text": review["text"],
                    "review_rating": review["rating"],
                })

    reviews_df = pd.DataFrame(review_records)
    print(f"Total reviews: {len(reviews_df):,}")
    if len(reviews_df) > 0:
        print(f"Reviews per movie (mean): {len(reviews_df) / len(df):.1f}")
        print(f"High-rated reviews (≥7/10): {(reviews_df['review_rating'] >= 7).sum():,}")

    # Save
    # Drop the raw reviews column from the main dataset (it's in the reviews file now)
    df = df.drop(columns=["reviews"])
    output_path = processed_dir / "movies_final.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\nSaved {len(df):,} movies to {output_path}")

    if len(reviews_df) > 0:
        reviews_path = processed_dir / "reviews.parquet"
        reviews_df.to_parquet(reviews_path, index=False)
        print(f"Saved {len(reviews_df):,} reviews to {reviews_path}")


if __name__ == "__main__":
    main()
