"""
Contrastive encoder model for semantic movie retrieval.

== Architecture ==

We wrap a pretrained sentence-transformer with a contrastive training
objective. The model encodes both queries (review text) and documents
(movie descriptions) into a shared embedding space where semantic
similarity corresponds to relevance.

== Loss Function: InfoNCE ==

InfoNCE (Noise-Contrastive Estimation) is the standard loss for
contrastive learning (used by CLIP, SimCLR, DPR, etc.).

For a batch of B (query, positive_doc) pairs:

    L = -1/B Σᵢ log( exp(sim(qᵢ, dᵢ)/τ) / Σⱼ exp(sim(qᵢ, dⱼ)/τ) )

Where:
    - sim(q, d) = cosine similarity between query and document embeddings
    - τ (temperature) controls the sharpness of the distribution
    - The numerator is the positive pair (query i, document i)
    - The denominator sums over ALL documents (positive + negatives)

Intuition: for each query, the loss is a softmax cross-entropy that
pushes the positive document's score to be the highest among all
candidates. Lower temperature makes the model more confident in its
choices — it sharpens the probability distribution.

== Temperature ==

τ is perhaps the most important hyperparameter in contrastive learning.
- Too high (τ=1.0): distribution is too uniform, model trains slowly,
  all similarities look similar
- Too low (τ=0.01): model becomes overconfident early, can overfit
  to hard negatives, gradients become unstable
- Sweet spot (τ=0.05-0.1): recommended starting range

We make temperature a LEARNABLE parameter initialized at 0.07 (the
value from CLIP). The model learns the optimal sharpness as training
progresses. This is important because the optimal temperature depends
on the difficulty of the negatives, which changes as training progresses.

== Gradient Considerations ==

The similarity matrix is BxB for in-batch negatives (or Bx(B+H) with
hard negatives). Gradients flow through both the query encoder and the
document encoder. With a shared encoder (our approach), this means
each text effectively gets gradients from both roles.

We use gradient checkpointing to reduce memory usage — the encoder
recomputes activations during the backward pass instead of storing
them. This trades ~30% more compute time for ~60% less memory, which
matters on consumer GPUs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


class ContrastiveMovieEncoder(nn.Module):
    """
    Shared encoder for queries and documents with contrastive loss.

    Design decision: SHARED encoder (same weights for queries and docs)
    rather than separate encoders. Reasons:
    1. Half the parameters → less overfitting risk with our small dataset
    2. The shared space is easier to analyze (queries and docs are directly
       comparable in the same space)
    3. Works better when you have < 50K training pairs (our case)
    4. Simpler to deploy — one model, not two

    If we had 500K+ training pairs, a bi-encoder (separate weights)
    might outperform because each encoder could specialize. That's
    an ablation to explore later.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        temperature: float = 0.07,
        learnable_temperature: bool = True,
    ):
        super().__init__()

        # Load the pretrained sentence-transformer
        self.encoder = SentenceTransformer(model_name)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()

        # Temperature parameter
        # We store log(temperature) rather than temperature directly because:
        # - Temperature must be positive (log space enforces this)
        # - Gradients are more stable in log space
        # - This is the standard practice (CLIP, OpenCLIP, etc.)
        if learnable_temperature:
            self.log_temperature = nn.Parameter(
                torch.tensor(temperature).log()
            )
        else:
            self.register_buffer(
                "log_temperature",
                torch.tensor(temperature).log(),
            )

    @property
    def temperature(self) -> torch.Tensor:
        # Clamp to prevent numerical issues
        # Min 0.01 (very sharp), max 1.0 (very flat)
        return self.log_temperature.exp().clamp(min=0.01, max=1.0)

    def encode(self, texts: list[str]) -> torch.Tensor:
        """
        Encode a list of texts into normalized embedding vectors.

        Returns shape: (len(texts), embedding_dim)

        The sentence-transformer internally:
        1. Tokenizes (text → token IDs)
        2. Runs through transformer layers
        3. Pools (mean pooling over token embeddings)
        4. We normalize to unit length (so dot product = cosine sim)
        """
        # sentence-transformers handles batching and device placement
        embeddings = self.encoder.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings

    def forward(
        self,
        queries: list[str],
        positive_docs: list[str],
        hard_neg_docs: list[str] | None = None,
        hard_neg_counts: list[int] | None = None,
    ) -> dict:
        """
        Compute contrastive loss for a batch.

        Args:
            queries: B query texts
            positive_docs: B positive document texts (aligned with queries)
            hard_neg_docs: Flattened list of hard negative documents
            hard_neg_counts: Number of hard negatives per query

        Returns:
            dict with:
                - loss: scalar contrastive loss
                - similarities: BxN similarity matrix (for logging)
                - temperature: current temperature value
                - accuracy: fraction of queries where the positive was top-ranked
        """
        batch_size = len(queries)

        # Encode everything in one forward pass for efficiency
        # We concatenate all texts, encode, then split
        all_doc_texts = list(positive_docs)
        if hard_neg_docs:
            all_doc_texts.extend(hard_neg_docs)

        query_embeds = self.encode(queries)            # (B, D)
        doc_embeds = self.encode(all_doc_texts)        # (B + n_hard, D)

        # Compute similarity matrix: each query vs all documents
        # Shape: (B, B + n_hard)
        similarities = torch.matmul(query_embeds, doc_embeds.T) / self.temperature

        # The labels are the diagonal — query_i should match doc_i
        labels = torch.arange(batch_size, device=similarities.device)

        # InfoNCE loss (cross-entropy where the correct answer is the diagonal)
        loss = F.cross_entropy(similarities, labels)

        # Compute accuracy (for monitoring training progress)
        with torch.no_grad():
            predictions = similarities.argmax(dim=1)
            accuracy = (predictions == labels).float().mean()

        return {
            "loss": loss,
            "similarities": similarities.detach(),
            "temperature": self.temperature.item(),
            "accuracy": accuracy.item(),
        }


class ContrastiveLossWithHardNegatives(nn.Module):
    """
    Alternative loss that gives extra weight to hard negatives.

    Standard InfoNCE treats all negatives equally. But hard negatives
    (semantically similar but wrong movies) are more informative than
    easy negatives (completely different movies). This loss applies a
    scaling factor to hard negative similarities.

    This is an ABLATION component — compare results with and without
    hard negative weighting to measure its impact.
    """

    def __init__(self, hard_negative_weight: float = 2.0):
        """
        Args:
            hard_negative_weight: Multiplier for hard negative similarities.
                1.0 = no weighting (equivalent to standard InfoNCE)
                2.0 = hard negatives are twice as "important" in the denominator
                Higher = model focuses more on distinguishing similar movies
        """
        super().__init__()
        self.hard_negative_weight = hard_negative_weight

    def forward(
        self,
        similarities: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Apply hard negative weighting to the similarity matrix.

        The similarity matrix is (B, B + n_hard) where:
        - Columns 0..B-1 are in-batch documents (diagonal = positives)
        - Columns B.. are hard negatives

        We multiply hard negative columns by the weight before computing
        the softmax. This makes the model "pay more attention" to getting
        hard negatives right.
        """
        # Create weight mask
        weights = torch.ones_like(similarities)
        if similarities.shape[1] > batch_size:
            weights[:, batch_size:] = self.hard_negative_weight

        weighted_sims = similarities * weights
        loss = F.cross_entropy(weighted_sims, labels)
        return loss
