"""
reranker.py -- CrossEncoder reranker for MedDRA candidate PTs.

Takes the Top-K candidates from HybridSearcher and scores each
(query, pt_name) pair with a CrossEncoder model, producing a more
precise relevance ranking than the bi-encoder used for retrieval.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~22MB)
  - MiniLM = distilled BERT, 6 transformer layers
  - Trained on MS MARCO passage retrieval (330M query-passage pairs)
  - Fast: ~1ms per pair on CPU
  - Good zero-shot transfer to clinical relevance scoring

Pipeline position:
  HybridSearcher (Top-20) --> CrossEncoder (reranked Top-5) --> LLM (final PT)

Usage:
  from vigilex.coding.reranker import CrossEncoderReranker
  from vigilex.coding.hybrid_search import HybridSearcher, EmbeddingModel

  reranker = CrossEncoderReranker()
  hybrid_results = searcher.search(query, top_k=20)
  reranked = reranker.rerank(query, hybrid_results, top_k=5)
"""

import sys
from dataclasses import dataclass
from typing import Optional

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    sys.exit(
        "sentence-transformers not installed. "
        "Run: pip3 install sentence-transformers --break-system-packages"
    )

from vigilex.coding.hybrid_search import SearchResult

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RerankedResult:
    pt_code:          int
    pt_name:          str
    soc_name:         str
    crossencoder_score: float       # raw CrossEncoder logit (higher = more relevant)
    rrf_score:        float         # original RRF score from hybrid search
    rrf_rank:         int           # original rank before reranking
    bm25_rank:        Optional[int]
    vector_rank:      Optional[int]
    trgm_sim:         Optional[float]
    cosine_sim:       Optional[float]


class CrossEncoderReranker:
    """
    Reranks hybrid search candidates using a CrossEncoder model.

    A CrossEncoder processes the query and candidate *together* in a single
    forward pass -- unlike the bi-encoder (PubMedBERT) which encodes them
    separately. This joint attention makes it significantly more accurate,
    at the cost of being slower (cannot pre-compute candidate embeddings).

    That is exactly why we use it as a *reranker* (on a small candidate set)
    rather than for initial retrieval (over all 27k PTs).

    Parameters
    ----------
    model_name : str
        HuggingFace model ID. Default: cross-encoder/ms-marco-MiniLM-L-6-v2
    max_length : int
        Max token length for (query, pt_name) pairs. 128 is more than enough
        since PT names are short (avg ~3 words).
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        max_length: int = 128,
    ):
        self.model = CrossEncoder(model_name, max_length=max_length)
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 5,
    ) -> list[RerankedResult]:
        """
        Score each (query, pt_name) pair and return top_k reranked results.

        Parameters
        ----------
        query : str
            The original free-text adverse event description.
        candidates : list[SearchResult]
            Output from HybridSearcher.search() -- typically top 20 results.
        top_k : int
            Number of results to return after reranking. Default: 5.

        Returns
        -------
        List of RerankedResult objects sorted by CrossEncoder score descending.
        """
        if not candidates:
            return []

        # Build (query, pt_name) pairs for CrossEncoder
        pairs = [(query, c.pt_name) for c in candidates]

        # Score all pairs in one batch -- returns array of relevance logits
        scores = self.model.predict(pairs)

        # Build reranked results
        reranked = []
        for i, (candidate, score) in enumerate(zip(candidates, scores)):
            reranked.append(RerankedResult(
                pt_code            = candidate.pt_code,
                pt_name            = candidate.pt_name,
                soc_name           = candidate.soc_name,
                crossencoder_score = float(score),
                rrf_score          = candidate.rrf_score,
                rrf_rank           = i + 1,
                bm25_rank          = candidate.bm25_rank,
                vector_rank        = candidate.vector_rank,
                trgm_sim           = candidate.trgm_sim,
                cosine_sim         = candidate.cosine_sim,
            ))

        reranked.sort(key=lambda x: x.crossencoder_score, reverse=True)
        return reranked[:top_k]
