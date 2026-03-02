"""
Evaluation framework for semantic movie retrieval.

This module computes standard information retrieval metrics that
measure how well the model ranks relevant movies for a given query.

== Why these specific metrics? ==

We evaluate at multiple levels because different metrics capture
different aspects of quality:

1. Hit Rate@K (Recall@K):
   "Is the correct movie in the top K results?"
   Binary — either it's there or it's not.
   Useful for: measuring basic retrieval capability.
   K=10 is standard because users rarely look past 10 results.

2. MRR (Mean Reciprocal Rank):
   "How high is the first correct result?"
   1/rank of the first correct answer, averaged across queries.
   MRR=1.0 means the correct movie is always #1.
   MRR=0.5 means it's typically #2.
   Useful for: measuring whether the model ranks the best result highly.

3. NDCG@K (Normalized Discounted Cumulative Gain):
   "How good is the overall ranking?"
   Accounts for position — a correct result at position 1 is worth more
   than at position 5. Logarithmic decay.
   Useful for: measuring full ranking quality, not just the top result.
   This is the most informative single metric for our use case.

== Evaluation Protocol ==

For each (query, positive_movie) pair in the eval set:
1. Encode the query
2. Retrieve top-K movies from the full FAISS index
3. Check if the positive movie appears in the results
4. Compute rank-based metrics

This is a "recall" evaluation — we're measuring whether the model
can find the right movie in a large catalog, not just rank a small
candidate set. This is harder and more realistic.

== Evaluation Splits ==

We evaluate on three axes to understand model behavior:

1. OVERALL: Average across all eval pairs
2. BY QUERY TYPE: Do short queries work as well as long ones?
   Do vibes-based queries work as well as plot-based queries?
3. BY MOVIE POPULARITY: Does the model work for obscure movies
   or only for popular ones? (A common failure mode)

== Comparing Models ==

All metrics include bootstrap confidence intervals so we can make
statistically rigorous claims about whether model A is better than
model B. Without CIs, a 0.5% improvement could be noise.
"""

import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def hit_rate_at_k(retrieved_ids: list[int], relevant_id: int, k: int) -> float:
    """
    Is the relevant movie in the top-K retrieved results?
    Returns 1.0 if yes, 0.0 if no.
    """
    return 1.0 if relevant_id in retrieved_ids[:k] else 0.0


def reciprocal_rank(retrieved_ids: list[int], relevant_id: int) -> float:
    """
    1 / (rank of first relevant result).
    Returns 0.0 if the relevant movie isn't in the retrieved list.
    """
    try:
        rank = retrieved_ids.index(relevant_id) + 1  # 1-indexed
        return 1.0 / rank
    except ValueError:
        return 0.0


def ndcg_at_k(retrieved_ids: list[int], relevant_id: int, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at K.

    For our case (one relevant document per query), this simplifies to:
        NDCG = 1 / log2(rank + 1) if relevant doc is in top-K, else 0

    The "normalized" part divides by the ideal DCG (which is 1/log2(2) = 1.0
    when the relevant doc is at position 1).
    """
    top_k = retrieved_ids[:k]
    if relevant_id not in top_k:
        return 0.0

    rank = top_k.index(relevant_id) + 1  # 1-indexed
    dcg = 1.0 / np.log2(rank + 1)
    ideal_dcg = 1.0 / np.log2(2)  # = 1.0 (relevant doc at position 1)
    return dcg / ideal_dcg


def bootstrap_confidence_interval(
    scores: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Why bootstrap over parametric CIs:
    - Retrieval metrics aren't normally distributed (they're bounded [0, 1]
      and often skewed)
    - Bootstrap makes no distributional assumptions
    - Standard in IR research (TREC uses this)

    Returns: (mean, lower_bound, upper_bound)
    """
    rng = np.random.RandomState(seed)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(scores, size=len(scores), replace=True)
        means.append(sample.mean())

    means = np.sort(means)
    alpha = 1 - confidence
    lower = means[int(alpha / 2 * n_bootstrap)]
    upper = means[int((1 - alpha / 2) * n_bootstrap)]
    return float(scores.mean()), float(lower), float(upper)


class RetrievalEvaluator:
    """
    Evaluates a model's retrieval quality on a held-out eval set.

    Usage:
        evaluator = RetrievalEvaluator(model, faiss_index, movie_ids, movies_df)
        results = evaluator.evaluate(eval_pairs_df, top_k_values=[5, 10, 20])
        evaluator.print_report(results)
    """

    def __init__(
        self,
        model: SentenceTransformer,
        faiss_index: faiss.Index,
        movie_ids: list[int],
        movies_df: pd.DataFrame | None = None,
    ):
        """
        Args:
            model: The encoder model (either baseline or fine-tuned)
            faiss_index: Pre-built FAISS index of movie embeddings
            movie_ids: tmdbId ordering matching the FAISS index rows
            movies_df: Full movie metadata (for stratified analysis)
        """
        self.model = model
        self.index = faiss_index
        self.movie_ids = movie_ids
        self.id_to_idx = {mid: i for i, mid in enumerate(movie_ids)}

        # Build popularity buckets for stratified evaluation
        self.popularity_bucket = {}
        if movies_df is not None:
            # Divide movies into quartiles by number of ratings
            quartiles = movies_df["num_ratings"].quantile([0.25, 0.5, 0.75])
            for _, row in movies_df.iterrows():
                n = row["num_ratings"]
                if n < quartiles[0.25]:
                    bucket = "rare"
                elif n < quartiles[0.5]:
                    bucket = "uncommon"
                elif n < quartiles[0.75]:
                    bucket = "common"
                else:
                    bucket = "popular"
                self.popularity_bucket[row["tmdbId"]] = bucket

    def retrieve(self, query_text: str, top_k: int = 100) -> list[int]:
        """Encode a query and retrieve top-K movie IDs."""
        if hasattr(self.model, 'encode'):
            query_vec = self.model.encode(
                [query_text],
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype(np.float32)
        else:
            # Handle our custom ContrastiveMovieEncoder
            with torch.no_grad():
                query_vec = self.model.encode([query_text]).cpu().numpy().astype(np.float32)

        if hasattr(self.index, "nprobe"):
            self.index.nprobe = 20

        scores, indices = self.index.search(query_vec, top_k)
        retrieved_ids = [self.movie_ids[i] for i in indices[0] if i >= 0]
        return retrieved_ids

    def evaluate(
        self,
        eval_df: pd.DataFrame,
        top_k_values: list[int] = [5, 10, 20],
        max_retrieve: int = 100,
    ) -> dict:
        """
        Run full evaluation on an eval set.

        Args:
            eval_df: DataFrame with columns [query_text, tmdb_id]
            top_k_values: Evaluate at these K values
            max_retrieve: Retrieve this many candidates per query

        Returns:
            Nested dict with overall metrics, per-K metrics,
            per-popularity-bucket metrics, and confidence intervals.
        """
        results = {
            "overall": {},
            "by_k": {},
            "by_popularity": {},
            "per_query": [],  # Raw per-query scores for analysis
        }

        all_mrr = []
        all_hit = {k: [] for k in top_k_values}
        all_ndcg = {k: [] for k in top_k_values}
        popularity_scores = {}

        for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Evaluating"):
            query = row["query_text"]
            relevant_id = int(row["tmdb_id"])

            # Skip if the relevant movie isn't in our index
            if relevant_id not in self.id_to_idx:
                continue

            retrieved = self.retrieve(query, top_k=max_retrieve)

            # Compute metrics
            mrr = reciprocal_rank(retrieved, relevant_id)
            all_mrr.append(mrr)

            query_result = {
                "query": query[:100],
                "relevant_id": relevant_id,
                "mrr": mrr,
            }

            for k in top_k_values:
                hr = hit_rate_at_k(retrieved, relevant_id, k)
                ndcg = ndcg_at_k(retrieved, relevant_id, k)
                all_hit[k].append(hr)
                all_ndcg[k].append(ndcg)
                query_result[f"hit@{k}"] = hr
                query_result[f"ndcg@{k}"] = ndcg

            results["per_query"].append(query_result)

            # Track by popularity bucket
            bucket = self.popularity_bucket.get(relevant_id, "unknown")
            if bucket not in popularity_scores:
                popularity_scores[bucket] = {"mrr": [], "ndcg@10": []}
            popularity_scores[bucket]["mrr"].append(mrr)
            if 10 in top_k_values:
                popularity_scores[bucket]["ndcg@10"].append(
                    ndcg_at_k(retrieved, relevant_id, 10)
                )

        # Aggregate overall metrics with confidence intervals
        mrr_arr = np.array(all_mrr)
        mrr_mean, mrr_lo, mrr_hi = bootstrap_confidence_interval(mrr_arr)
        results["overall"]["mrr"] = {
            "mean": mrr_mean, "ci_lower": mrr_lo, "ci_upper": mrr_hi
        }

        for k in top_k_values:
            hit_arr = np.array(all_hit[k])
            ndcg_arr = np.array(all_ndcg[k])

            hit_mean, hit_lo, hit_hi = bootstrap_confidence_interval(hit_arr)
            ndcg_mean, ndcg_lo, ndcg_hi = bootstrap_confidence_interval(ndcg_arr)

            results["by_k"][k] = {
                "hit_rate": {"mean": hit_mean, "ci_lower": hit_lo, "ci_upper": hit_hi},
                "ndcg": {"mean": ndcg_mean, "ci_lower": ndcg_lo, "ci_upper": ndcg_hi},
            }

        # Per-popularity-bucket breakdown
        for bucket, scores in popularity_scores.items():
            if len(scores["mrr"]) > 10:  # Only report if enough samples
                mrr_arr = np.array(scores["mrr"])
                results["by_popularity"][bucket] = {
                    "n_queries": len(mrr_arr),
                    "mrr": float(mrr_arr.mean()),
                    "ndcg@10": float(np.array(scores["ndcg@10"]).mean()) if scores["ndcg@10"] else 0,
                }

        results["n_evaluated"] = len(all_mrr)
        return results

    @staticmethod
    def print_report(results: dict, model_name: str = "Model"):
        """Pretty-print evaluation results."""
        print(f"\n{'=' * 70}")
        print(f"EVALUATION REPORT: {model_name}")
        print(f"{'=' * 70}")
        print(f"Evaluated on {results['n_evaluated']:,} query-movie pairs\n")

        # Overall MRR
        mrr = results["overall"]["mrr"]
        print(f"MRR: {mrr['mean']:.4f}  [{mrr['ci_lower']:.4f}, {mrr['ci_upper']:.4f}]")

        # Per-K metrics
        print(f"\n{'K':>4}  {'Hit Rate':>12}  {'95% CI':>16}  {'NDCG':>8}  {'95% CI':>16}")
        print("-" * 65)
        for k, metrics in sorted(results["by_k"].items()):
            hr = metrics["hit_rate"]
            ndcg = metrics["ndcg"]
            print(
                f"{k:>4}  {hr['mean']:>12.4f}  "
                f"[{hr['ci_lower']:.4f}, {hr['ci_upper']:.4f}]  "
                f"{ndcg['mean']:>8.4f}  "
                f"[{ndcg['ci_lower']:.4f}, {ndcg['ci_upper']:.4f}]"
            )

        # Popularity breakdown
        if results["by_popularity"]:
            print(f"\nPerformance by movie popularity:")
            print(f"{'Bucket':>10}  {'N queries':>10}  {'MRR':>8}  {'NDCG@10':>8}")
            print("-" * 45)
            for bucket in ["rare", "uncommon", "common", "popular"]:
                if bucket in results["by_popularity"]:
                    b = results["by_popularity"][bucket]
                    print(
                        f"{bucket:>10}  {b['n_queries']:>10}  "
                        f"{b['mrr']:>8.4f}  {b['ndcg@10']:>8.4f}"
                    )


def compare_models(
    results_a: dict,
    results_b: dict,
    name_a: str = "Baseline",
    name_b: str = "Fine-tuned",
    metric: str = "ndcg",
    k: int = 10,
):
    """
    Statistical comparison between two models.

    Uses paired bootstrap test: for each bootstrap sample, compute the
    difference in metrics between model A and B. If the 95% CI of the
    difference excludes zero, the improvement is statistically significant.

    This is the standard way to compare IR systems (used in TREC evaluations).
    """
    scores_a = np.array([q[f"{metric}@{k}"] for q in results_a["per_query"]])
    scores_b = np.array([q[f"{metric}@{k}"] for q in results_b["per_query"]])

    # Paired bootstrap
    n_bootstrap = 5000
    rng = np.random.RandomState(42)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(scores_a), size=len(scores_a), replace=True)
        diff = scores_b[idx].mean() - scores_a[idx].mean()
        diffs.append(diff)

    diffs = np.sort(diffs)
    mean_diff = np.mean(diffs)
    ci_lower = diffs[int(0.025 * n_bootstrap)]
    ci_upper = diffs[int(0.975 * n_bootstrap)]

    # p-value: fraction of bootstrap samples where B is not better than A
    p_value = (np.array(diffs) <= 0).mean()

    print(f"\n{'=' * 70}")
    print(f"MODEL COMPARISON: {name_a} vs {name_b}")
    print(f"{'=' * 70}")
    print(f"Metric: {metric}@{k}")
    print(f"{name_a}: {scores_a.mean():.4f}")
    print(f"{name_b}: {scores_b.mean():.4f}")
    print(f"Difference: {mean_diff:+.4f}  [{ci_lower:+.4f}, {ci_upper:+.4f}]")
    print(f"p-value: {p_value:.4f}")

    if ci_lower > 0:
        print(f"→ {name_b} is SIGNIFICANTLY better (p={p_value:.4f})")
    elif ci_upper < 0:
        print(f"→ {name_a} is SIGNIFICANTLY better (p={1-p_value:.4f})")
    else:
        print(f"→ Difference is NOT statistically significant")
