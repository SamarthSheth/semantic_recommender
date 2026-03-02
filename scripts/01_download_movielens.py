"""
Step 1: Download and process MovieLens data.

What this does:
    1. Downloads MovieLens 25M dataset (or uses cached version)
    2. Filters to movies with enough ratings (configured in config.yaml)
    3. Extracts the TMDB IDs for those movies (this is our join key)
    4. Computes per-movie rating statistics
    5. Saves a clean parquet file of movies we want to fetch from TMDB

Why MovieLens first:
    MovieLens is our quality filter. TMDB has metadata for ~800K movies,
    but most are obscure films with no ratings. By starting with MovieLens,
    we identify the ~10K movies that real users have actually watched and
    rated, then only fetch TMDB data for those. This saves thousands of
    API calls and gives us a higher-quality catalog.

Output: data/processed/movielens_filtered.parquet
    Columns: movieId, title, genres, tmdbId, num_ratings, avg_rating, rating_std
"""

import zipfile
import io
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from config_loader import load_config


MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"


def download_movielens(raw_dir: str) -> Path:
    """
    Download MovieLens 25M if not already cached.

    Decision: download the full zip rather than individual files because
    GroupLens distributes it as a single archive. We extract only the
    files we need (ratings.csv, movies.csv, links.csv) to save disk space.
    """
    extract_dir = Path(raw_dir) / "ml-25m"

    if extract_dir.exists() and (extract_dir / "ratings.csv").exists():
        print(f"MovieLens already downloaded at {extract_dir}")
        return extract_dir

    print(f"Downloading MovieLens 25M (~250MB)...")
    response = requests.get(MOVIELENS_URL, stream=True)
    response.raise_for_status()

    # Stream into memory then extract — avoids writing a temp zip file
    total_size = int(response.headers.get("content-length", 0))
    buffer = io.BytesIO()
    with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            buffer.write(chunk)
            pbar.update(len(chunk))

    print("Extracting...")
    buffer.seek(0)
    with zipfile.ZipFile(buffer) as zf:
        # Only extract what we need
        needed_files = ["ml-25m/ratings.csv", "ml-25m/movies.csv", "ml-25m/links.csv"]
        for name in needed_files:
            zf.extract(name, raw_dir)

    return extract_dir


def process_movielens(extract_dir: Path, config: dict) -> pd.DataFrame:
    """
    Filter and aggregate MovieLens data.

    Key decisions:
    - We aggregate ratings per movie (not per user) because our unit of
      analysis is a movie, not a user. We're building a catalog, not a
      collaborative filter (yet).
    - We keep rating_std because it captures polarization: a movie with
      avg=3.5 and std=0.5 is "meh" while avg=3.5 and std=1.5 is
      polarizing (some love it, some hate it). This could be a useful
      feature later.
    """
    min_ratings = config["data"]["min_ratings_per_movie"]
    min_avg = config["data"]["min_avg_rating"]

    print("Loading ratings...")
    # dtype optimization: movieId and userId as int32 saves ~50% memory
    # on 25M rows this matters (ratings.csv is ~650MB)
    ratings = pd.read_csv(
        extract_dir / "ratings.csv",
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
        usecols=["userId", "movieId", "rating", "timestamp"],
    )
    print(f"  Loaded {len(ratings):,} ratings")

    # Aggregate per movie
    movie_stats = (
        ratings.groupby("movieId")
        .agg(
            num_ratings=("rating", "count"),
            avg_rating=("rating", "mean"),
            rating_std=("rating", "std"),
            # Keep latest timestamp for temporal splitting later
            latest_rating=("timestamp", "max"),
        )
        .reset_index()
    )

    # Filter
    before = len(movie_stats)
    movie_stats = movie_stats[
        (movie_stats["num_ratings"] >= min_ratings)
        & (movie_stats["avg_rating"] >= min_avg)
    ]
    print(f"  Filtered {before:,} → {len(movie_stats):,} movies "
          f"(min_ratings={min_ratings}, min_avg={min_avg})")

    # Join with movie metadata (title, genres)
    movies = pd.read_csv(extract_dir / "movies.csv")
    movie_stats = movie_stats.merge(movies, on="movieId", how="left")

    # Join with links to get TMDB IDs
    # Decision: drop movies without a TMDB ID — we can't fetch metadata for them
    links = pd.read_csv(extract_dir / "links.csv")
    movie_stats = movie_stats.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
    before = len(movie_stats)
    movie_stats = movie_stats.dropna(subset=["tmdbId"])
    movie_stats["tmdbId"] = movie_stats["tmdbId"].astype(int)
    print(f"  Dropped {before - len(movie_stats)} movies without TMDB IDs "
          f"→ {len(movie_stats):,} remaining")

    return movie_stats


def main():
    config = load_config()

    # Step 1: Download
    extract_dir = download_movielens(config["paths"]["raw_data"])

    # Step 2: Process
    df = process_movielens(extract_dir, config)

    # Step 3: Save as parquet
    # Decision: parquet over CSV because:
    # - Preserves dtypes (no re-parsing int vs float on reload)
    # - Columnar compression (~3-5x smaller than CSV)
    # - Faster reads when you only need a subset of columns
    output_path = Path(config["paths"]["processed_data"]) / "movielens_filtered.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\nSaved {len(df):,} movies to {output_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSample:\n{df.head()}")


if __name__ == "__main__":
    main()
