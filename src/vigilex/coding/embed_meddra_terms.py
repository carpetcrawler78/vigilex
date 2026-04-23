"""
embed_meddra_terms.py -- Generate PubMedBERT embeddings for all MedDRA PT names
and store them in processed.meddra_terms.pt_embedding (VECTOR(768)).

Model: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract (~440 MB)
Runtime: ~10-20 min on CPU for 27k terms (one-time job)

Usage (from vigilex repo root):
    python3 -m src.vigilex.coding.embed_meddra_terms
    python3 -m src.vigilex.coding.embed_meddra_terms --batch-size 64 --db-url postgresql://...

Prerequisites:
    pip3 install transformers torch psycopg2-binary python-dotenv --break-system-packages

After this script completes, run the SQL to build the IVFFlat index:
    docker exec vigilex-postgres psql -U vigilex -d vigilex -c "
        CREATE INDEX idx_meddra_embedding
        ON processed.meddra_terms
        USING ivfflat (pt_embedding vector_cosine_ops)
        WITH (lists = 100);"
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

MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
EMBEDDING_DIM = 768


def get_db_url() -> str:
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
    """Fetch all PT codes and names that have no embedding yet."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT pt_code, pt_name
            FROM processed.meddra_terms
            WHERE pt_embedding IS NULL
            ORDER BY pt_code
        """)
        return cur.fetchall()


def mean_pooling(model_output, attention_mask):
    """Mean pooling over token embeddings (standard for sentence embeddings)."""
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1).clamp(min=1e-9)


def embed_texts(texts: list[str], tokenizer, model, device) -> list[list[float]]:
    """Encode a batch of texts to 768-dim vectors."""
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=64,   # PT names are short -- 64 tokens is more than enough
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output = model(**encoded)

    embeddings = mean_pooling(output, encoded["attention_mask"])
    # L2-normalize for cosine similarity via pgvector
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().tolist()


def update_embeddings(conn, pt_codes: list[int], embeddings: list[list[float]]) -> None:
    """Write embeddings back to processed.meddra_terms."""
    sql = """
        UPDATE processed.meddra_terms
        SET pt_embedding = %s::vector
        WHERE pt_code = %s
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(
            cur,
            sql,
            [(str(emb), code) for code, emb in zip(pt_codes, embeddings)],
            page_size=200
        )
    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PubMedBERT embeddings for MedDRA PT names"
    )
    parser.add_argument("--db-url", default=None)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Texts per forward pass (default: 32, reduce if OOM)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load model and encode first batch only, no DB writes")
    args = parser.parse_args()

    # --- DB connection ---
    db_url = args.db_url or get_db_url()
    print("Connecting to DB...")
    try:
        conn = psycopg2.connect(db_url)
    except psycopg2.OperationalError as e:
        sys.exit(f"DB connection failed: {e}")

    # --- Fetch PTs without embeddings ---
    rows = load_pt_names(conn)
    if not rows:
        print("All PT embeddings already populated. Nothing to do.")
        conn.close()
        return

    print(f"PTs to embed: {len(rows):,}")

    # --- Load model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading model: {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    print("Model loaded.\n")

    # --- Batch embedding ---
    batch_size = args.batch_size
    total = len(rows)
    start_time = time.time()

    for i in range(0, total, batch_size):
        batch = rows[i:i + batch_size]
        pt_codes = [r[0] for r in batch]
        pt_names = [r[1] for r in batch]

        embeddings = embed_texts(pt_names, tokenizer, model, device)

        if not args.dry_run:
            update_embeddings(conn, pt_codes, embeddings)

        elapsed = time.time() - start_time
        done = i + len(batch)
        pct = done / total * 100
        eta = (elapsed / done) * (total - done) if done > 0 else 0
        print(f"  [{done:>6,}/{total:,}] {pct:5.1f}%  elapsed: {elapsed:5.0f}s  ETA: {eta:5.0f}s",
              end="\r")

        if args.dry_run and i == 0:
            print(f"\n[dry-run] First batch encoded OK. Shape: {len(embeddings)} x {len(embeddings[0])}")
            print("[dry-run] No DB writes performed.")
            conn.close()
            return

    print(f"\n\nDone. {total:,} embeddings written in {time.time()-start_time:.0f}s")
    print("\nNext step: build the IVFFlat index:")
    print("  docker exec vigilex-postgres psql -U vigilex -d vigilex -c \\")
    print('    "CREATE INDEX idx_meddra_embedding ON processed.meddra_terms')
    print('     USING ivfflat (pt_embedding vector_cosine_ops) WITH (lists = 100);"')

    conn.close()


if __name__ == "__main__":
    main()
