"""
PyTorch Dataset for contrastive learning on movie recommendations.

This dataset serves (query, positive_document, hard_negative_documents)
triplets to the training loop. The design choices here affect training
speed, memory usage, and ultimately model quality.

== How contrastive batches work ==

A batch of size B contains:
    - B query texts (review chunks)
    - B positive document texts (the movie each review is about)
    - Optionally, B×H hard negative document texts

The training loss uses in-batch negatives: for query_i, the positive
is document_i, and the negatives are document_j for all j ≠ i.
Hard negatives are additional negatives that are close in embedding
space (and therefore harder to distinguish).

== Why not just use sentence-transformers' built-in trainer? ==

We could use `sentence_transformers.SentenceTransformer.fit()` with
`InputExample` objects. But writing our own loop gives us:
1. Full control over hard negative sampling strategy
2. Custom logging and evaluation hooks
3. The ability to implement the same loop in JAX for benchmarking
4. It demonstrates deeper understanding than calling a library
"""

import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class ContrastiveMovieDataset(Dataset):
    """
    Dataset that yields (query, positive_doc, hard_negative_docs) tuples.

    Each item returns raw text strings — tokenization happens in the
    training loop via the model's tokenizer. This is a deliberate
    choice: keeping tokenization in the dataloader would couple the
    dataset to a specific model, making it harder to swap models
    during ablation experiments.
    """

    def __init__(
        self,
        pairs_path: str | Path,
        movie_docs: dict[int, str],
        n_hard_negatives: int = 4,
    ):
        """
        Args:
            pairs_path: Path to parquet file from prepare_training_data.py
            movie_docs: Dict mapping tmdbId → document text
            n_hard_negatives: Number of hard negatives per query.
                More = harder training signal but more memory per batch.
                We default to 4 because:
                - With batch_size=64, you already get 63 in-batch negatives
                - 4 hard negatives add the most informative ones on top
                - More than ~8 has diminishing returns and slows training
        """
        self.pairs = pd.read_parquet(pairs_path)
        self.movie_docs = movie_docs
        self.n_hard_negatives = n_hard_negatives

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx) -> dict:
        row = self.pairs.iloc[idx]

        query = row["query_text"]
        positive_doc = row["document_text"]

        # Parse hard negative IDs and look up their documents
        hard_neg_ids = json.loads(row["hard_negative_ids"])
        hard_neg_docs = []
        for neg_id in hard_neg_ids[: self.n_hard_negatives]:
            if neg_id in self.movie_docs:
                hard_neg_docs.append(self.movie_docs[neg_id])

        return {
            "query": query,
            "positive_doc": positive_doc,
            "hard_neg_docs": hard_neg_docs,
            "tmdb_id": int(row["tmdb_id"]),
        }


def collate_contrastive(batch: list[dict]) -> dict:
    """
    Custom collate function for contrastive batches.

    Standard PyTorch collation doesn't handle variable-length lists
    of hard negatives well. This function:
    1. Separates queries and positive docs into flat lists (for batch encoding)
    2. Flattens hard negatives into a single list with index tracking
       so we can reassemble them after encoding.

    Why flatten: The encoder processes all texts in one forward pass,
    which is much faster than encoding queries, positives, and negatives
    separately. We track which embedding belongs to which role via indices.
    """
    queries = [item["query"] for item in batch]
    positive_docs = [item["positive_doc"] for item in batch]
    tmdb_ids = [item["tmdb_id"] for item in batch]

    # Flatten hard negatives and track boundaries
    all_hard_negs = []
    hard_neg_counts = []  # how many hard negatives each query has
    for item in batch:
        negs = item["hard_neg_docs"]
        all_hard_negs.extend(negs)
        hard_neg_counts.append(len(negs))

    return {
        "queries": queries,
        "positive_docs": positive_docs,
        "hard_neg_docs": all_hard_negs,
        "hard_neg_counts": hard_neg_counts,
        "tmdb_ids": tmdb_ids,
    }


def create_dataloaders(
    train_path: str | Path,
    val_path: str | Path,
    movie_docs: dict[int, str],
    batch_size: int = 64,
    n_hard_negatives: int = 4,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    batch_size=64 rationale:
    - 64 queries × 64 positives = 4096-entry similarity matrix
    - Each query gets 63 in-batch negatives + 4 hard negatives = 67 total
    - This is a strong learning signal without requiring huge GPU memory
    - Larger batches (128, 256) give more negatives but need more VRAM
      and can actually hurt when training data is small (our case)

    num_workers=0:
    - Our dataset is small enough that the overhead of multiprocessing
      isn't worth it. The bottleneck is the model forward pass, not data loading.
    - On macOS, multiprocessing with PyTorch can cause issues.
    """
    train_dataset = ContrastiveMovieDataset(
        train_path, movie_docs, n_hard_negatives
    )
    val_dataset = ContrastiveMovieDataset(
        val_path, movie_docs, n_hard_negatives
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,       # Random order each epoch — important for
                            # in-batch negatives to vary across epochs
        collate_fn=collate_contrastive,
        num_workers=num_workers,
        drop_last=True,     # Drop incomplete last batch because in-batch
                            # negatives need consistent batch size
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_contrastive,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, val_loader
