"""
embed_meddra_terms.py -- Generate PubMedBERT embeddings for all MedDRA PT names.

What is an embedding?
    An embedding is a list of numbers (a vector) that represents the meaning of
    a piece of text in a mathematical space. Texts with similar meanings have
    vectors that point in similar directions (high cosine similarity).

    For example:
        "Hypoglycaemia"         -> [0.12, -0.45, 0.88, ...]  (768 numbers)
        "Low blood glucose"     -> [0.11, -0.44, 0.85, ...]  (similar!)
        "Ventricular fibrillation" -> [-0.33, 0.71, -0.22, ...] (very different)

    This allows us to find relevant MedDRA terms even when the exact words
    don't match -- a report saying "patient's blood sugar dropped dangerously"
    would have an embedding close to "Hypoglycaemia" even though the word
    "hypoglycaemia" never appears.

Why PubMedBERT?
    PubMedBERT (microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract) was
    pre-trained exclusively on PubMed abstracts. Unlike general-purpose BERT,
    it understands biomedical terminology natively. It produces much better
    embeddings for medical text than models trained on Wikipedia or news.

    Model size: ~440 MB (downloaded from Hugging Face on first run)
    Embedding dimension: 768 numbers per text
    Runtime: ~10-20 minutes on CPU for 27,361 PT names (one-time job)

What does this script do?
    1. Connects to PostgreSQL and finds all PT rows without an embedding yet.
    2. Loads the PubMedBERT model (downloads if not cached).
    3. Encodes all PT names in batches (32 names per forward pass).
    4. Writes the 768-dimensional vectors back to the pt_embedding column.

After this script, you must build the IVFFlat index:
    docker exec vigilex-postgres psql -U vigilex -d vigilex -c "
        CREATE INDEX idx_meddra_embedding
        ON processed.meddra_terms
        USING ivfflat (pt_embedding vector_cosine_ops)
        WITH (lists = 100);"

Usage:
    python3 -m src.vigilex.coding.embed_meddra_terms
    python3 -m src.vigilex.coding.embed_meddra_terms --batch-size 64
    python3 -m src.vigilex.coding.embed_meddra_terms --dry-run  # test without writing
"""

import argparse
import os
import sys
import time
from pathlib import Path

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    sys.exit("psycopg2-binary not installed. Run: pip3 install psycopg2-binary --break-system-packages")

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    sys.exit("transformers/torch not installed. Run: pip3 install transformers torch --break-system-packages")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# The specific PubMedBERT model checkpoint on Hugging Face Hub.
# It was pre-trained on 14 million PubMed abstracts using masked language
# modeling (the same BERT pre-training objective, but on biomedical text).
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

# The number of dimensions in each embedding vector.
# BERT-base models always produce 768-dimensional vectors.
# This must match the VECTOR(768) type in the PostgreSQL schema.
EMBEDDING_DIM = 768


def get_db_url() -> str:
    """
    Return the PostgreSQL connection URL from environment variables.
    Prefers DATABASE_URL (docker-compose convention); falls back to
    individual POSTGRES_* variables for local development.
    """
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db   = os.getenv("POSTGRES_DB", "vigilex")
    user = os.getenv("POSTGRES_USER", "vigilex")
    pw   = os.getenv("POSTGRES_PASSWORD", "")
    return f"postgresql://{user}:{pw}@{host}:{port}/{db}"


def load_pt_names(conn) -> list[tuple[int, str]]:
    """
    Fetch all PT codes and names that do not yet have an embedding.

    The WHERE pt_embedding IS NULL clause means the script is safe to re-run:
    if it was interrupted halfway, only the remaining PTs are processed.
    PTs that already have embeddings are skipped.

    Returns:
        List of (pt_code, pt_name) tuples ordered by pt_code.
        Example: [(10018429, "Hypoglycaemia"), (10047065, "Ventricular tachycardia"), ...]
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT pt_code, pt_name
            FROM processed.meddra_terms
            WHERE pt_embedding IS NULL
            ORDER BY pt_code
        """)
        return cur.fetchall()


def mean_pooling(model_output, attention_mask):
    """
    Compute a single sentence embedding by averaging token embeddings.

    What is mean pooling?
        BERT produces one 768-dimensional vector per token (word piece),
        not per sentence. To get a single vector representing the whole text,
        we average over all token vectors, weighted by the attention mask
        (so we do not include padding tokens in the average).

    Why not use the [CLS] token embedding instead?
        The [CLS] token (first token, special to BERT) was designed for
        classification tasks. Mean pooling over all tokens consistently
        produces better sentence-level embeddings for semantic similarity.

    Args:
        model_output:   Output from the transformer model (contains last_hidden_state)
        attention_mask: Binary mask (1 for real tokens, 0 for padding tokens)

    Returns:
        Tensor of shape (batch_size, 768) -- one embedding per input text.
    """
    token_embeddings = model_output.last_hidden_state  # shape: (batch, seq_len, 768)
    # Expand the mask to match the embedding dimension
    input_mask_expanded = (
        attention_mask
        .unsqueeze(-1)                                   # shape: (batch, seq_len, 1)
        .expand(token_embeddings.size())                 # shape: (batch, seq_len, 768)
        .float()
    )
    # Sum the token embeddings weighted by the mask, then divide by the mask sum.
    # clamp(min=1e-9) prevents division by zero for empty inputs.
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1).clamp(min=1e-9)


def embed_texts(texts: list[str], tokenizer, model, device) -> list[list[float]]:
    """
    Encode a batch of text strings into 768-dimensional embedding vectors.

    Steps:
        1. Tokenize: convert text to token IDs (subwords the model understands)
        2. Forward pass: run the transformer (no gradient computation needed)
        3. Mean pooling: collapse token embeddings to one sentence embedding
        4. L2 normalise: make all vectors unit length for cosine similarity

    Why L2 normalise?
        pgvector's cosine distance operator (<=>)  measures 1 - cosine_similarity.
        Cosine similarity of two unit vectors equals their dot product. Normalising
        during embedding generation means we can use the fast dot product operator
        instead of the full cosine formula at query time.

    Why max_length=64?
        MedDRA PT names are very short (average 3-5 words). 64 tokens is far
        more than enough. Using a smaller max_length makes each forward pass
        faster and uses less memory.

    Args:
        texts:     List of PT name strings to encode.
        tokenizer: Hugging Face tokenizer for PubMedBERT.
        model:     PubMedBERT model in eval mode.
        device:    "cuda" or "cpu".

    Returns:
        List of 768-dimensional float lists (one per input text).
    """
    encoded = tokenizer(
        texts,
        padding=True,       # pad shorter texts in the batch to equal length
        truncation=True,    # truncate texts longer than max_length (rare for PT names)
        max_length=64,
        return_tensors="pt" # return PyTorch tensors (not numpy or plain lists)
    ).to(device)

    with torch.no_grad():  # no_grad = skip gradient computation (we're not training)
        output = model(**encoded)

    embeddings = mean_pooling(output, encoded["attention_mask"])
    # L2 normalise along dimension 1 (the embedding dimension)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().tolist()  # move to CPU and convert to plain Python lists


def update_embeddings(conn, pt_codes: list[int], embeddings: list[list[float]]) -> None:
    """
    Write the generated embeddings back to the processed.meddra_terms table.

    The embedding is stored as a PostgreSQL VECTOR(768) type (from pgvector).
    We cast the Python list to a string first, then cast it to ::vector in SQL
    (e.g. "[0.12, -0.45, ...]"::vector).

    Args:
        conn:       open psycopg2 connection
        pt_codes:   list of integer PT codes
        embeddings: list of 768-dim float lists (one per pt_code)
    """
    sql = """
        UPDATE processed.meddra_terms
        SET pt_embedding = %s::vector
        WHERE pt_code = %s
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(
            cur,
            sql,
            # Pair each embedding (as string) with its pt_code
            [(str(emb), code) for code, emb in zip(pt_codes, embeddings)],
            page_size=200
        )
    conn.commit()


def main() -> None:
    """
    Main entry point: load model, encode all unembedded PTs, write to DB.
    """
    parser = argparse.ArgumentParser(
        description="Generate PubMedBERT embeddings for MedDRA PT names"
    )
    parser.add_argument(
        "--db-url", default=None,
        help="PostgreSQL URL (default: reads from DATABASE_URL environment variable)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Number of PT names to encode per model forward pass (default: 32). "
             "Reduce if you run out of memory (OOM error)."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load the model and encode the first batch, but do not write to DB. "
             "Useful for verifying the model loads correctly without a DB connection."
    )
    args = parser.parse_args()

    # --- Connect to database ---
    db_url = args.db_url or get_db_url()
    print("Connecting to database...")
    try:
        conn = psycopg2.connect(db_url)
    except psycopg2.OperationalError as e:
        sys.exit(f"Database connection failed: {e}")

    # --- Find PTs that still need embeddings ---
    rows = load_pt_names(conn)
    if not rows:
        print("All PT embeddings already populated. Nothing to do.")
        conn.close()
        return

    print(f"PTs to embed: {len(rows):,}")

    # --- Load the PubMedBERT model ---
    # Use GPU (CUDA) if available, otherwise CPU.
    # On the Hetzner CX33 server (no GPU), this runs on CPU (~10-20 min for 27k PTs).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading model: {MODEL_NAME} ...")
    print("(First run downloads ~440 MB from Hugging Face -- subsequent runs use cache)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()  # eval mode: disables dropout, batch norm behaves differently -- important!
    print("Model loaded.\n")

    # --- Encode in batches ---
    batch_size = args.batch_size
    total      = len(rows)
    start_time = time.time()

    for i in range(0, total, batch_size):
        batch    = rows[i : i + batch_size]
        pt_codes = [r[0] for r in batch]
        pt_names = [r[1] for r in batch]

        embeddings = embed_texts(pt_names, tokenizer, model, device)

        if not args.dry_run:
            update_embeddings(conn, pt_codes, embeddings)

        # Progress display (overwrites same line with \r)
        elapsed = time.time() - start_time
        done    = i + len(batch)
        pct     = done / total * 100
        eta     = (elapsed / done) * (total - done) if done > 0 else 0
        print(
            f"  [{done:>6,}/{total:,}]  {pct:5.1f}%  "
            f"elapsed: {elapsed:5.0f}s  ETA: {eta:5.0f}s",
            end="\r"
        )

        if args.dry_run and i == 0:
            print(f"\n[dry-run] First batch OK. Shape: {len(embeddings)} x {len(embeddings[0])}")
            print("[dry-run] No database writes performed.")
            conn.close()
            return

    print(f"\n\nDone. {total:,} embeddings written in {time.time()-start_time:.0f}s")
    print("\nNext step: build the IVFFlat index for fast approximate nearest-neighbour search:")
    print("  docker exec vigilex-postgres psql -U vigilex -d vigilex -c \\")
    print('    "CREATE INDEX idx_meddra_embedding ON processed.meddra_terms')
    print('     USING ivfflat (pt_embedding vector_cosine_ops) WITH (lists = 100);"')

    conn.close()


if __name__ == "__main__":
    main()
