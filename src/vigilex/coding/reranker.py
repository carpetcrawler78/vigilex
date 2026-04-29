"""
reranker.py -- CrossEncoder reranker for MedDRA candidate PTs.

Why do we need a reranker at all?
    The hybrid search (BM25 + vector) is fast but imprecise. It retrieves 20
    candidates efficiently, but the ranking is based on approximate signals:
    trigram overlap and embedding similarity. These signals do not actually
    read the query and the candidate *together* -- they compare them separately.

    The CrossEncoder is much more accurate because it processes the full
    (query, candidate) pair jointly through a neural network. This joint
    attention lets the model understand relationships like:
        - "Did the narrative describe THIS specific condition or just mention it?"
        - "Is the PT name a description of a cause or an outcome?"
        - "Are there any negations? ('denied experiencing hypoglycaemia')"

The two-stage pipeline design:
    Retrieval (fast, approximate) -> Reranking (slow, precise)

    Stage 1 (HybridSearcher):        27,361 PTs  ->  Top-20 candidates  (fast)
    Stage 2 (CrossEncoderReranker):  Top-20      ->  Top-5  candidates  (precise)
    Stage 3 (LLMCoder):              Top-5       ->  Final PT code       (interpretable)

    Why not run the CrossEncoder over all 27,361 PTs directly?
        The CrossEncoder scores one (query, PT) pair per forward pass.
        27,361 pairs * ~1ms each = ~27 seconds per report. That is too slow.
        Running it over only 20 candidates takes ~20ms -- acceptable.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    - MiniLM: a distilled (smaller, faster) version of BERT with 6 transformer layers
    - Trained on MS MARCO: a large dataset of 330 million real web search query/passage pairs
    - Despite being trained on web search (not clinical data), it transfers well to
      relevance ranking in the medical domain -- it understands "does this text
      describe this concept?" regardless of domain

This module was explored and validated in Notebook 06_meddra_reranker.ipynb.
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

# The CrossEncoder model from Hugging Face Hub (~22 MB -- much smaller than PubMedBERT)
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ---------------------------------------------------------------------------
# Data class for reranked results
# ---------------------------------------------------------------------------

@dataclass
class RerankedResult:
    """
    One candidate MedDRA PT after CrossEncoder reranking.

    Carries all the information from the hybrid search stage (for traceability)
    plus the CrossEncoder relevance score that determined the new ranking.
    """
    pt_code:             int
    pt_name:             str
    soc_name:            str
    crossencoder_score:  float        # raw CrossEncoder output (logit, not probability)
                                      # higher = more relevant; can be any real number
    rrf_score:           float        # original RRF score from hybrid search
    rrf_rank:            int          # rank before reranking (1 = best hybrid result)
    bm25_rank:           Optional[int]
    vector_rank:         Optional[int]
    trgm_sim:            Optional[float]
    cosine_sim:          Optional[float]


# ---------------------------------------------------------------------------
# CrossEncoder Reranker
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """
    Reranks hybrid search candidates using a CrossEncoder neural model.

    Bi-encoder vs CrossEncoder -- the key distinction:
        Bi-encoder (PubMedBERT in hybrid_search.py):
            Encodes the query separately: embed(query)
            Encodes each candidate separately: embed(PT_name)
            Score = cosine_similarity(embed(query), embed(PT_name))
            Fast because PT embeddings are pre-computed and cached in the DB.
            Less accurate because query and PT never interact directly.

        CrossEncoder (this module):
            Encodes the pair together: score(query, PT_name)
            The full transformer sees query AND PT_name simultaneously.
            Every attention head can attend from query tokens to PT tokens and back.
            Much more accurate, but cannot pre-compute -- must run fresh per query.

        This trade-off is why we use the bi-encoder for retrieval (over 27k PTs)
        and the CrossEncoder for reranking (over 20 candidates only).

    Parameters:
        model_name: Hugging Face model ID. Default: cross-encoder/ms-marco-MiniLM-L-6-v2
        max_length: Maximum token length for (query + pt_name) combined.
                    128 is more than enough since PT names are very short.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        max_length: int = 128,
    ):
        # CrossEncoder from sentence-transformers handles the model loading,
        # tokenisation, and inference -- much simpler than using transformers directly.
        self.model = CrossEncoder(model_name, max_length=max_length)
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 5,
    ) -> list[RerankedResult]:
        """
        Score each (query, pt_name) pair and return the top_k reranked results.

        The CrossEncoder outputs a single relevance score (logit) for each pair.
        This is not a probability (it can be negative or > 1) -- it is a raw
        "relevance score" where higher values indicate more relevant matches.
        In the next stage (LLMCoder), we convert this to a probability using
        sigmoid() before incorporating it into the final confidence score.

        Args:
            query:      The original MAUDE narrative text.
            candidates: List of SearchResult objects from HybridSearcher.search().
                        Typically the top 20 hybrid results.
            top_k:      Number of results to return. Default: 5 (sent to LLMCoder).

        Returns:
            List of RerankedResult sorted by crossencoder_score descending (best first).
        """
        if not candidates:
            return []

        # Build (query, pt_name) pairs for batch scoring.
        # The CrossEncoder processes all pairs in a single efficient batch call.
        # Note: we use the full query here (not just the first sentence) because
        # the CrossEncoder is fast and can handle longer inputs -- the full context
        # helps it distinguish between primary and secondary adverse events.
        pairs = [(query, c.pt_name) for c in candidates]

        # Score all pairs in one forward pass.
        # Returns a numpy array of shape (len(candidates),) with one float per pair.
        scores = self.model.predict(pairs)

        # Build RerankedResult objects preserving the full score history
        reranked = []
        for i, (candidate, score) in enumerate(zip(candidates, scores)):
            reranked.append(RerankedResult(
                pt_code            = candidate.pt_code,
                pt_name            = candidate.pt_name,
                soc_name           = candidate.soc_name,
                crossencoder_score = float(score),       # the new ranking criterion
                rrf_score          = candidate.rrf_score, # original rank from hybrid
                rrf_rank           = i + 1,               # 1 = was best hybrid result
                bm25_rank          = candidate.bm25_rank,
                vector_rank        = candidate.vector_rank,
                trgm_sim           = candidate.trgm_sim,
                cosine_sim         = candidate.cosine_sim,
            ))

        # Sort by CrossEncoder score (highest = most relevant) and return top_k
        reranked.sort(key=lambda x: x.crossencoder_score, reverse=True)
        return reranked[:top_k]
