"""
Step 2: I'll Fetch movie metadata and reviews from the TMDB API.

What this does:
    1. Reads the filtered movie list from step 1
    2. For each movie, fetches: overview, genres, keywords, reviews in a single API call
    3. Checkpoints progress every 500 movies (resume-safe)
    4. Saves full TMDB metadata as parquet

Why we need this step:
    MovieLens gives us ratings but almost no text — just titles and broad
    genre labels like "Action|Comedy". For embedding-based retrieval, we
    need rich text descriptions. TMDB provides:
    - Plot overviews (the core text we'll embed)
    - Fine-grained keywords ("time travel", "dystopia", "based on novel")
    - User reviews (training signal for contrastive learning later)
    - Structured metadata (cast, crew, release year)

Prerequisites:
    - Run 01_download_movielens.py first
    - Set TMDB_API_KEY environment variable (get a free key at themoviedb.org)

Output: data/processed/tmdb_metadata.parquet
    Columns: tmdbId, overview, genres_tmdb, keywords, cast, director,
             release_year, popularity, vote_average, reviews
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from config_loader import load_config


def get_api_key() -> str:
    """
    Read TMDB API key from environment.

    Decision: env var over config file because API keys should never
    be committed to git. An env var also works cleanly in CI/CD and
    Docker without mounting secret files.
    """
    key = os.environ.get("TMDB_API_KEY") #my TMDB API key was saved here
    if not key:
        raise EnvironmentError(
            "Set TMDB_API_KEY environment variable.\n"
            "Get a free key at: https://www.themoviedb.org/settings/api"
        )
    return key


def fetch_movie_details(tmdb_id: int, api_key: str) -> dict | None:
    """
    Fetch metadata for a single movie.

    We use the 'append_to_response' parameter to batch multiple
    sub-requests (keywords, credits, reviews) into a single API call.
    This is 1 call instead of 4, which is critical for staying under
    rate limits when fetching 10K+ movies.
    """
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
    params = {
        "api_key": api_key,
        "append_to_response": "keywords,credits,reviews",
        "language": "en-US",
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 404:
            return None  # Movie removed from TMDB
        if resp.status_code == 429:
            # Rate limited — back off and retry
            retry_after = int(resp.headers.get("Retry-After", 2))
            time.sleep(retry_after)
            return fetch_movie_details(tmdb_id, api_key)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"  Error fetching tmdb_id={tmdb_id}: {e}")
        return None


def parse_movie_response(data: dict, max_reviews: int) -> dict:
    """
    Extract the fields we care about from the raw TMDB response.

    Decision: flatten the nested JSON into a flat dict here rather than
    storing raw JSON. Reasons:
    - Raw TMDB responses are ~5-10KB each (cast lists are huge).
      10K movies × 10KB = 100MB of mostly unused data.
    - Flat structure loads directly into a DataFrame without post-processing.
    - We extract exactly what we need for embeddings and metadata.

    What we extract and why:
    - overview: primary text for embedding. This is the "about" for each movie.
    - keywords: thematic tags like "dystopia", "revenge". These capture
      abstract themes that overviews sometimes miss.
    - genres: structured category labels. Useful for evaluation (does the
      model cluster genres correctly?) and as metadata features.
    - director + top cast: people are a strong signal. "A Wes Anderson film"
      or "starring Daniel Day-Lewis" carries meaning.
    - reviews: training data for contrastive learning (step 4). A review
      like "hauntingly beautiful slow burn" paired with this movie teaches
      the model to map vibes-language to content.
    """
    # Extract director from credits
    director = None
    if "credits" in data and "crew" in data["credits"]:
        directors = [c["name"] for c in data["credits"]["crew"] if c["job"] == "Director"]
        director = directors[0] if directors else None

    # Extract top-billed cast (first 5)
    cast = []
    if "credits" in data and "cast" in data["credits"]:
        cast = [c["name"] for c in data["credits"]["cast"][:5]]

    # Extract keywords
    keywords = []
    if "keywords" in data and "keywords" in data["keywords"]:
        keywords = [k["name"] for k in data["keywords"]["keywords"]]

    # Extract genres
    genres = [g["name"] for g in data.get("genres", [])]

    # Extract reviews (text + rating)
    reviews = []
    if "reviews" in data and "results" in data["reviews"]:
        for review in data["reviews"]["results"][:max_reviews]:
            reviews.append({
                "text": review.get("content", ""),
                "rating": review.get("author_details", {}).get("rating"),
            })

    return {
        "tmdbId": data["id"],
        "title_tmdb": data.get("title", ""),
        "overview": data.get("overview", ""),
        "genres_tmdb": genres,
        "keywords": keywords,
        "director": director,
        "cast": cast,
        "release_date": data.get("release_date", ""),
        "popularity": data.get("popularity", 0),
        "vote_average": data.get("vote_average", 0),
        "vote_count": data.get("vote_count", 0),
        "reviews": reviews,
    }


def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load previously fetched results to resume from where we left off."""
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            data = json.load(f)
        print(f"Resuming from checkpoint: {len(data)} movies already fetched")
        return data
    return {}


def save_checkpoint(results: dict, checkpoint_path: Path):
    """
    Save progress to disk.

    Decision: JSON checkpoint over parquet because we're writing
    incrementally. Parquet requires rewriting the entire file each time,
    JSON lets us just dump the dict. We convert to parquet at the end.
    """
    with open(checkpoint_path, "w") as f:
        json.dump(results, f)


def main():
    config = load_config()
    api_key = get_api_key()

    # Load filtered movie list from step 1
    movies_path = Path(config["paths"]["processed_data"]) / "movielens_filtered.parquet"
    if not movies_path.exists():
        raise FileNotFoundError(
            f"{movies_path} not found. Run 01_download_movielens.py first."
        )
    movies = pd.read_parquet(movies_path)
    tmdb_ids = movies["tmdbId"].tolist()
    print(f"Need to fetch metadata for {len(tmdb_ids):,} movies")

    # Setup checkpointing
    checkpoint_path = Path(config["paths"]["raw_data"]) / "tmdb_checkpoint.json"
    results = load_checkpoint(checkpoint_path)
    already_fetched = set(results.keys())

    # Fetch
    max_reviews = config["data"]["tmdb_max_reviews_per_movie"]
    delay = config["data"]["tmdb_rate_limit_delay"]
    checkpoint_every = 500

    remaining = [tid for tid in tmdb_ids if str(tid) not in already_fetched]
    print(f"Remaining to fetch: {len(remaining):,}")

    for i, tmdb_id in enumerate(tqdm(remaining, desc="Fetching TMDB")):
        raw = fetch_movie_details(tmdb_id, api_key)
        if raw is not None:
            parsed = parse_movie_response(raw, max_reviews)
            results[str(tmdb_id)] = parsed

        # Checkpoint periodically
        if (i + 1) % checkpoint_every == 0:
            save_checkpoint(results, checkpoint_path)
            tqdm.write(f"  Checkpointed at {len(results):,} movies")

        time.sleep(delay)

    # Final save
    save_checkpoint(results, checkpoint_path)

    # Convert to DataFrame and save as parquet
    # Decision: store list fields (genres, keywords, cast, reviews) as
    # JSON strings in parquet. Parquet doesn't handle nested lists well
    # across all readers. JSON strings are universally parseable.
    records = list(results.values())
    df = pd.DataFrame(records)
    for col in ["genres_tmdb", "keywords", "cast", "reviews"]:
        df[col] = df[col].apply(json.dumps)

    output_path = Path(config["paths"]["processed_data"]) / "tmdb_metadata.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\nSaved {len(df):,} movies to {output_path}")

    # Quick stats
    has_overview = (df["overview"].str.len() > 0).sum()
    has_reviews = (df["reviews"] != "[]").sum()
    print(f"Movies with overviews: {has_overview:,}")
    print(f"Movies with reviews: {has_reviews:,}")


if __name__ == "__main__":
    main()
