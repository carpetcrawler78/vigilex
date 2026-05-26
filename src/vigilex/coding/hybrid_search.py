"""
hybrid_search.py -- Hybrid BM25 + Vector search over MedDRA terms.

The core challenge of MedDRA coding:
    A MAUDE narrative might say: "patient experienced a sudden drop in blood sugar
    requiring emergency glucagon injection after the pump delivered an unintended bolus."
    The correct MedDRA PT is "Hypoglycaemia". The word "hypoglycaemia" never appears
    in the narrative -- we need to infer it from context.

Why two retrieval methods?
    No single method handles all cases well:

    BM25 (lexical/keyword search):
        + Fast, exact term matching
        + Works well when the narrative uses MedDRA vocabulary or LLT synonyms
        - Fails completely when different words mean the same thing

    Vector search (semantic/embedding search):
        + Understands meaning even when words differ (e.g. "low blood sugar" ~ "hypoglycaemia")
        + Robust to paraphrasing and clinical abbreviations
        - Can retrieve semantically similar but clinically wrong terms

    Hybrid (BM25 + Vector):
        = Gets the best of both worlds -- catches exact matches AND semantic matches

Fusion method -- Reciprocal Rank Fusion (RRF):
    Instead of combining raw scores (which are on different scales and hard to
    normalise), we combine the RANKS. For each candidate PT that appears in
    either result list:

        RRF_score(PT) = w_bm25   / (k + rank_bm25(PT))
                      + w_vector / (k + rank_vector(PT))

    Where k=60 is a smoothing constant that dampens differences at the top of
    the rankings (e.g. rank 1 vs rank 2 matters less than rank 1 vs rank 100).

    Weights: w_bm25=0.4, w_vector=0.6
    We weight vector higher because PT names use clinical vocabulary that often
    differs significantly from narrative language.

This implementation was developed and validated in Notebooks 05 and 06.
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
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit("sentence-transformers not installed. Run: pip3 install sentence-transformers")


# ---------------------------------------------------------------------------
# RRF hyperparameters (tuned during development, see Notebook 05)
# ---------------------------------------------------------------------------

# k=60 is the standard RRF constant from the original paper (Cormack et al. 2009).
# It means the score contribution of rank-1 is 1/(60+1)=0.016 vs rank-2 = 1/(60+2)=0.016
# The difference is small, which prevents a single method from dominating too much.
RRF_K        = 60

# BM25 (trigram) gets less weight because MedDRA PT names use standardised clinical
# terminology that often differs from the colloquial language in MAUDE narratives.
RRF_W_BM25   = 0.4

# Vector search gets more weight because all-mpnet-base-v2 captures semantic meaning
# across vocabulary differences better than lexical matching alone.
RRF_W_VECTOR = 0.6

# all-mpnet-base-v2: chosen over PubMedBERT after bench 2026-05-26 (R@5: 0.25 vs 0.0).
# Must match the model used in embed_meddra_terms_v2.py -- query and index must share
# the same embedding space.
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


# ---------------------------------------------------------------------------
# Data classes for structured results
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """
    One candidate MedDRA Preferred Term from the hybrid search.

    Contains both the fused RRF score (used for ranking) and the raw
    scores from each individual retrieval arm (useful for debugging and
    for understanding where a match came from).
    """
    pt_code:      int
    pt_name:      str
    soc_name:     str           # e.g. "Metabolism and nutrition disorders"
    rrf_score:    float         # fused score (higher = better match)
    bm25_rank:    Optional[int] # rank in the BM25 result list (None if not retrieved)
    vector_rank:  Optional[int] # rank in the vector result list (None if not retrieved)
    trgm_sim:     Optional[float] # raw trigram similarity score (0.0-1.0)
    cosine_sim:   Optional[float] # raw cosine similarity (0.0-1.0 after L2 norm)


# ---------------------------------------------------------------------------
# Embedding model wrapper
# ---------------------------------------------------------------------------

class EmbeddingModel:
    """
    Wrapper around all-mpnet-base-v2 (SentenceTransformer) for query encoding.

    Replaces the previous PubMedBERT implementation (embed_meddra_terms.py).
    Bench 2026-05-26 showed all-mpnet-base-v2 + pt_only achieves R@5=0.25 vs
    PubMedBERT R@5=0.0 on the same production index.

    Must use the same model as embed_meddra_terms_v2.py -- query and index
    vectors must share the same embedding space.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self._model = SentenceTransformer(model_name)

    def encode(self, text: str) -> list[float]:
        """Encode text into a normalised 768-dim vector (L2 unit length)."""
        # normalize_embeddings=True: <=> works without it, but L2 normalization
        # makes cosine distance more robust and consistent -- good practice,
        # not a hard requirement from pgvector.
        return self._model.encode(
            text,
            normalize_embeddings=True,
        ).tolist()


# ---------------------------------------------------------------------------
# Hybrid Searcher
# ---------------------------------------------------------------------------

class HybridSearcher:
    """
    Hybrid BM25 + Vector searcher over processed.meddra_terms.

    Architecture overview:
        query text
            |
            +--> BM25 arm (pg_trgm)    --> top-100 candidates by trigram similarity
            |
            +--> Vector arm (pgvector) --> top-100 candidates by cosine similarity
                    |
                    v (encode with PubMedBERT first)
            |
            +--> RRF fusion --> top-K results by combined rank

    Parameters:
        conn:            psycopg2 connection (replaced per batch for self-healing)
        embedding_model: EmbeddingModel instance (loaded once, reused for all queries)
        candidate_pool:  number of candidates to fetch from each retrieval arm.
                         More candidates = better recall but slower fusion.
        rrf_k:           RRF smoothing constant (see module docstring)
    """

    def __init__(
        self,
        conn,
        embedding_model: Optional[EmbeddingModel] = None,
        candidate_pool: int = 100,
        rrf_k: float = RRF_K,
    ):
        self.conn = conn
        # Load the embedding model if one was not provided (e.g. during testing)
        self.model = embedding_model or EmbeddingModel()
        self.candidate_pool = candidate_pool
        self.rrf_k = rrf_k

    # ------------------------------------------------------------------
    # BM25 arm: lexical search via pg_trgm trigrams
    # ------------------------------------------------------------------

    def _bm25_search(self, query: str) -> list[dict]:
        """
        Retrieve top candidates by trigram (character 3-gram) similarity.

        What is a trigram?
            PostgreSQL pg_trgm splits text into overlapping 3-character sequences
            called trigrams. For example, "heart" -> {" he", "hea", "ear", "art", "rt "}.
            The similarity between two strings is the Jaccard similarity of their
            trigram sets: how many trigrams do they share?

        Why word_similarity() instead of similarity()?
            similarity(a, b)      measures overlap of the full trigram sets.
            word_similarity(a, b) measures how well `a` appears as a contiguous
                                  word sequence within `b`.

            We use word_similarity(pt_name, query_narrative) because pt_name is
            short (e.g. "Hypoglycaemia") and query is long. Regular similarity
            would penalise pt_name for not covering all trigrams in the long query.
            This was Bug #3 in Notebook 08.

        Searches both:
            - pt_name (direct PT match)
            - llt_name (synonym match -- catches e.g. "Low blood glucose" for Hypoglycaemia)

        Returns rows sorted by similarity score descending.
        """
        sql = """
            WITH pt_sim AS (
                -- Direct PT name match.
                -- lower() on both sides ensures case-insensitive matching
                -- (MedDRA PTs use title case; narratives use mixed case).
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
                -- LLT synonym match.
                -- Joins the synonym (llt_name) against the narrative,
                -- then returns the parent PT. This way a synonym match
                -- surfaces the corresponding PT, not the LLT itself.
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
                -- Union both match types
                SELECT * FROM pt_sim
                UNION ALL
                SELECT * FROM llt_sim
            )
            -- A PT might appear via both its name and a synonym.
            -- Group by pt_code and keep the highest similarity score.
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
    # Vector arm: semantic search via pgvector cosine similarity
    # ------------------------------------------------------------------

    def _vector_search(self, query_embedding: list[float], limit: int = 0) -> list[dict]:
        """
        Retrieve top candidates by cosine similarity to the query embedding.

        Uses embedding_mpnet (all-mpnet-base-v2, 768-dim) and idx_meddra_mpnet_ivfflat.
        Called twice per search() invocation (query fusion): once with first_sentence,
        once with full_text_truncated. Results are merged and deduped before RRF fusion.

        Args:
            query_embedding: L2-normalised 768-dim vector from EmbeddingModel.encode().
            limit: max rows to return. Defaults to self.candidate_pool if 0.
        """
        n = limit if limit > 0 else self.candidate_pool
        sql = """
            SELECT
                pt_code,
                pt_name,
                soc_name,
                1 - (embedding_mpnet <=> %(embedding)s::vector) AS cosine_sim
            FROM processed.meddra_terms
            WHERE embedding_mpnet IS NOT NULL
            ORDER BY embedding_mpnet <=> %(embedding)s::vector
            LIMIT %(limit)s
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Session-level setting: search all 100 IVFFlat clusters (exact retrieval).
            # Lower probes (e.g. 10) would be faster but approximate -- keep at 100
            # until production quality is validated.
            cur.execute("SET ivfflat.probes = 100")
            cur.execute(sql, {"embedding": str(query_embedding), "limit": n})
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
        Fuse BM25 and vector result lists using Weighted Reciprocal Rank Fusion.

        Algorithm:
            1. For each unique PT code in either result list, compute:
               RRF_score = w_bm25 / (k + bm25_rank)  [if in bm25 list]
                         + w_vector / (k + vector_rank) [if in vector list]
            2. If a PT appears in only one list, it still gets a positive score
               from that one arm (the other arm contributes 0).
            3. Sort all PTs by RRF score descending and return the top_k.

        Example:
            PT "Hypoglycaemia":
                BM25 rank  = 1   -> contribution: 0.4 / (60+1)  = 0.00656
                Vector rank= 1   -> contribution: 0.6 / (60+1)  = 0.00984
                Total RRF  = 0.01640  (appears in both -> highest possible score)

            PT "Glucose disorder NEC":
                BM25 rank  = 5   -> contribution: 0.4 / (60+5)  = 0.00615
                Vector rank= None (not retrieved) -> 0
                Total RRF  = 0.00615
        """
        # Build rank lookup dicts: pt_code -> rank (1-indexed, so rank 1 = best)
        bm25_rank   = {r["pt_code"]: i + 1 for i, r in enumerate(bm25_results)}
        vector_rank = {r["pt_code"]: i + 1 for i, r in enumerate(vector_results)}

        # Build raw score lookup dicts for reporting purposes
        bm25_sim   = {r["pt_code"]: r["trgm_sim"]  for r in bm25_results}
        vector_sim = {r["pt_code"]: r["cosine_sim"] for r in vector_results}

        # Metadata (name, SOC) -- take from either list; same PT has same metadata
        pt_meta = {r["pt_code"]: r for r in bm25_results + vector_results}

        # Union of all retrieved PT codes (from either arm)
        all_pt_codes = set(bm25_rank.keys()) | set(vector_rank.keys())

        scores = []
        for pt_code in all_pt_codes:
            rrf = 0.0
            # Add BM25 contribution if this PT appeared in the BM25 results
            if pt_code in bm25_rank:
                rrf += RRF_W_BM25 / (self.rrf_k + bm25_rank[pt_code])
            # Add vector contribution if this PT appeared in the vector results
            if pt_code in vector_rank:
                rrf += RRF_W_VECTOR / (self.rrf_k + vector_rank[pt_code])

            meta = pt_meta[pt_code]
            scores.append(SearchResult(
                pt_code     = pt_code,
                pt_name     = meta["pt_name"],
                soc_name    = meta["soc_name"],
                rrf_score   = rrf,
                bm25_rank   = bm25_rank.get(pt_code),    # None if not in BM25 results
                vector_rank = vector_rank.get(pt_code),  # None if not in vector results
                trgm_sim    = bm25_sim.get(pt_code),
                cosine_sim  = vector_sim.get(pt_code),
            ))

        # Sort by RRF score (highest = best match) and return the top_k results
        scores.sort(key=lambda x: x.rrf_score, reverse=True)
        return scores[:top_k]

    # ------------------------------------------------------------------
    # Public search interface
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """
        Search MedDRA terms for a given free-text query.

        This is the main function called by the CodingWorker for each MAUDE report.
        It orchestrates the two retrieval arms and the RRF fusion.

        Design decision -- first sentence only:
            We encode only the FIRST SENTENCE of the narrative for the vector arm.
            MAUDE narratives can be several paragraphs long. When we encode the
            full narrative, the embedding becomes a "bag of topics" that includes
            device descriptions, manufacturer information, and contextual detail.
            This dilutes the clinical signal (Bug #5 in Notebook 08).
            The first sentence typically states the primary adverse event directly.

            Example:
                Full narrative: "Patient experienced hypoglycemia requiring emergency
                glucagon injection. The Omnipod 5 device had been calibrated correctly..."
                First sentence: "Patient experienced hypoglycemia requiring emergency
                glucagon injection."

            The first-sentence encoding is far more useful for finding "Hypoglycaemia".

        Args:
            query:  Free-text adverse event description (MAUDE mdr_text excerpt).
            top_k:  Number of results to return after fusion. Default: 10.
                    In the full pipeline, top_k=20 (then reranked to Top-5).

        Returns:
            List of SearchResult objects sorted by RRF score descending.
        """
        # Query Fusion: two complementary query representations for the vector arm.
        # Bench 2026-05-26 showed these are complementary:
        #   first_sentence -> better precision at top ranks (R@1/R@5)
        #   full_text_truncated -> better coverage at depth (R@100)
        # Running both and unioning candidates captures the strengths of each.
        q1 = query.split(".")[0].strip() or query   # first sentence
        q2 = query[:512]                             # truncated full text

        emb1 = self.model.encode(q1)
        emb2 = self.model.encode(q2)

        # Fetch top-50 from each query (union gives up to 100 unique candidates)
        c1 = self._vector_search(emb1, limit=50)
        c2 = self._vector_search(emb2, limit=50)

        # Deduplicate by pt_code, keeping the row with the higher cosine_sim
        seen: dict = {}
        for row in c1 + c2:
            code = row["pt_code"]
            if code not in seen or row["cosine_sim"] > seen[code]["cosine_sim"]:
                seen[code] = row
        vector_results = sorted(seen.values(), key=lambda x: x["cosine_sim"], reverse=True)

        # BM25 arm: unchanged
        bm25_results = self._bm25_search(query)

        return self._rrf_fuse(bm25_results, vector_results, top_k)
