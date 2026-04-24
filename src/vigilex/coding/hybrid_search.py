"""
hybrid_search.py -- Hybrid BM25 + Vector search over MedDRA terms.

Given a free-text MAUDE adverse event narrative, returns the top-K most
relevant MedDRA Preferred Terms (PTs) by fusing two ranked lists:

  1. BM25 (lexical)  -- pg_trgm trigram similarity on pt_name + llt_name
  2. Vector (semantic) -- pgvector cosine similarity on PubMedBERT embeddings

Fusion method: Weighted Reciprocal Rank Fusion (RRF)
  score_rrf(d) = sum( w_i / (k + rank_i(d)) )

  k = 60  (standard RRF constant -- dampens rank differences at the top)
  w_bm25   = 0.4  (lexical signal)
  w_vector = 0.6  (semantic signal -- higher because PT names are clinical,
                   surface form often differs from MAUDE narrative vocabulary)

Usage:
  from vigilex.coding.hybrid_search import HybridSearcher
  searcher = HybridSearcher(conn)
  results = searcher.search("patient experienced hypoglycemia after bolus delivery", top_k=10)
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    sys.exit("psycopg2-binary not installed.")

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    sys.exit("transformers/torch not installed.")


# ---------------------------------------------------------------------------
# RRF hyperparameters
# ---------------------------------------------------------------------------
RRF_K        = 60    # standard constant -- reduces sensitivity to top rank gaps
RRF_W_BM25   = 0.4  # weight for lexical (trigram) results
RRF_W_VECTOR = 0.6  # weight for semantic (embedding) results

MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"


@dataclass
class SearchResult:
    pt_code:      int
    pt_name:      str
    soc_name:     str
    rrf_score:    float
    bm25_rank:    Optional[int]   # rank in BM25 results (None if not retrieved)
    vector_rank:  Optional[int]   # rank in vector results (None if not retrieved)
    trgm_sim:     Optional[float] # raw trigram similarity score
    cosine_sim:   Optional[float] # raw cosine similarity score


class EmbeddingModel:
    """Lightweight wrapper around PubMedBERT for query encoding."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, text: str) -> list[float]:
        """Encode a single text to a normalized 768-dim vector."""
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self.model(**encoded)

        # Mean pooling over token embeddings
        token_emb = output.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(token_emb.size()).float()
        pooled = (token_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        # L2 normalize
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return normalized[0].cpu().tolist()


class HybridSearcher:
    """
    Hybrid BM25 + Vector searcher over processed.meddra_terms.

    Parameters
    ----------
    conn : psycopg2 connection
    embedding_model : EmbeddingModel instance (shared across calls for efficiency)
    candidate_pool : int
        How many candidates to retrieve from each retrieval arm before fusion.
        More candidates = better recall, slower query. Default: 100.
    rrf_k : float
        RRF constant. Higher = less aggressive rank weighting. Default: 60.
    """

    def __init__(
        self,
        conn,
        embedding_model: Optional[EmbeddingModel] = None,
        candidate_pool: int = 100,
        rrf_k: float = RRF_K,
    ):
        self.conn = conn
        self.model = embedding_model or EmbeddingModel()
        self.candidate_pool = candidate_pool
        self.rrf_k = rrf_k

    # ------------------------------------------------------------------
    # BM25 arm: trigram similarity via pg_trgm
    # ------------------------------------------------------------------
    def _bm25_search(self, query: str) -> list[dict]:
        """
        Retrieve top candidates by trigram similarity.

        Searches both pt_name (direct PT match) and llt_name (synonym match)
        to handle cases where the MAUDE narrative uses a non-preferred term.

        Returns rows ordered by similarity descending.
        """
        sql = """
            WITH pt_sim AS (
                -- word_similarity with lower() on both sides for case-insensitive matching.
                -- word_similarity(pt_name, query): how well does pt_name appear as a
                -- contiguous word sequence inside the (longer) query narrative?
                SELECT
                    t.pt_code,
                    t.pt_name,
                    t.soc_name,
                    word_similarity(lower(t.pt_name), lower(%(query)s)) AS sim
                FROM processed.meddra_terms t
                WHERE word_similarity(lower(t.pt_name), lower(%(query)s)) > 0.1
                ORDER BY sim DESC
                LIMIT %(limit)s
            ),
            llt_sim AS (
                -- LLT synonym similarity -> join back to PT (also lowercased)
                SELECT
                    t.pt_code,
                    t.pt_name,
                    t.soc_name,
                    word_similarity(lower(l.llt_name), lower(%(query)s)) AS sim
                FROM processed.meddra_llt l
                JOIN processed.meddra_terms t USING (pt_code)
                WHERE word_similarity(lower(l.llt_name), lower(%(query)s)) > 0.1
                ORDER BY sim DESC
                LIMIT %(limit)s
            ),
            combined AS (
                SELECT * FROM pt_sim
                UNION ALL
                SELECT * FROM llt_sim
            )
            SELECT pt_code, pt_name, soc_name, MAX(sim) AS trgm_sim
            FROM combined
            GROUP BY pt_code, pt_name, soc_name
            ORDER BY trgm_sim DESC
            LIMIT %(limit)s
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, {"query": query, "limit": self.candidate_pool})
            return cur.fetchall()

    # ------------------------------------------------------------------
    # Vector arm: cosine similarity via pgvector
    # ------------------------------------------------------------------
    def _vector_search(self, query_embedding: list[float]) -> list[dict]:
        """
        Retrieve top candidates by cosine similarity to query embedding.

        Uses the IVFFlat index on pt_embedding for fast ANN search.
        pgvector <=> operator = cosine distance (1 - cosine_similarity).
        """
        sql = """
            SELECT
                pt_code,
                pt_name,
                soc_name,
                1 - (pt_embedding <=> %(embedding)s::vector) AS cosine_sim
            FROM processed.meddra_terms
            WHERE pt_embedding IS NOT NULL
            ORDER BY pt_embedding <=> %(embedding)s::vector
            LIMIT %(limit)s
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # probes=100 = full scan through all clusters (exact results)
            # Lower to 10-20 in production once index quality is verified
            cur.execute("SET ivfflat.probes = 100")
            cur.execute(sql, {
                "embedding": str(query_embedding),
                "limit": self.candidate_pool
            })
            return cur.fetchall()

    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------
    def _rrf_fuse(
        self,
        bm25_results: list[dict],
        vector_results: list[dict],
        top_k: int,
    ) -> list[SearchResult]:
        """
        Weighted Reciprocal Rank Fusion.

        For each document d in the union of both result lists:
            score(d) = w_bm25   / (k + rank_bm25(d))
                     + w_vector / (k + rank_vector(d))

        Documents not retrieved by an arm get rank = infinity (no contribution).
        """
        # Build rank lookup dicts: pt_code -> rank (1-indexed)
        bm25_rank   = {r["pt_code"]: i + 1 for i, r in enumerate(bm25_results)}
        vector_rank = {r["pt_code"]: i + 1 for i, r in enumerate(vector_results)}

        # Raw score lookups
        bm25_sim   = {r["pt_code"]: r["trgm_sim"]  for r in bm25_results}
        vector_sim = {r["pt_code"]: r["cosine_sim"] for r in vector_results}
        pt_meta    = {r["pt_code"]: r for r in bm25_results + vector_results}

        # Union of all retrieved PT codes
        all_pt_codes = set(bm25_rank.keys()) | set(vector_rank.keys())

        scores = []
        for pt_code in all_pt_codes:
            rrf = 0.0
            if pt_code in bm25_rank:
                rrf += RRF_W_BM25 / (self.rrf_k + bm25_rank[pt_code])
            if pt_code in vector_rank:
                rrf += RRF_W_VECTOR / (self.rrf_k + vector_rank[pt_code])

            meta = pt_meta[pt_code]
            scores.append(SearchResult(
                pt_code     = pt_code,
                pt_name     = meta["pt_name"],
                soc_name    = meta["soc_name"],
                rrf_score   = rrf,
                bm25_rank   = bm25_rank.get(pt_code),
                vector_rank = vector_rank.get(pt_code),
                trgm_sim    = bm25_sim.get(pt_code),
                cosine_sim  = vector_sim.get(pt_code),
            ))

        scores.sort(key=lambda x: x.rrf_score, reverse=True)
        return scores[:top_k]

    # ------------------------------------------------------------------
    # Public search interface
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """
        Search MedDRA terms for a given free-text query.

        Parameters
        ----------
        query : str
            Free-text adverse event description (e.g. MAUDE narrative excerpt)
        top_k : int
            Number of results to return after fusion. Default: 10.

        Returns
        -------
        List of SearchResult objects sorted by RRF score descending.
        """
        # Encode query for vector arm.
        # Use only the first sentence -- long narratives produce diluted embeddings
        # where device/procedure terms dominate over clinical outcomes.
        # The first sentence typically contains the primary adverse event.
        first_sentence = query.split(".")[0].strip()
        query_embedding = self.model.encode(first_sentence if first_sentence else query)

        # Run both arms
        bm25_results   = self._bm25_search(query)
        vector_results = self._vector_search(query_embedding)

        # Fuse and return
        return self._rrf_fuse(bm25_results, vector_results, top_k)
