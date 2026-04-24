"""
Diagnose: sind lokale Query-Embeddings kompatibel mit den in der DB gespeicherten?

Laedt das Hypoglycaemia-PT-Embedding aus der DB und vergleicht es:
  1. Mit sich selbst (sollte similarity ~1.0 sein)
  2. Mit einem lokal generierten Embedding von "Hypoglycaemia"
  3. Mit einem lokal generierten Embedding der Test-Narrative

Usage: python scripts/diagnose_embeddings.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from dotenv import load_dotenv
load_dotenv()

import psycopg2
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

def get_db_url():
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db   = os.getenv("POSTGRES_DB", "vigilex")
    user = os.getenv("POSTGRES_USER", "vigilex")
    pw   = os.getenv("POSTGRES_PASSWORD", "")
    return f"postgresql://{user}:{pw}@{host}:{port}/{db}"

def encode(text, tokenizer, model):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    with torch.no_grad():
        out = model(**enc)
    token_emb = out.last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).expand(token_emb.size()).float()
    pooled = (token_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return torch.nn.functional.normalize(pooled, p=2, dim=1)[0].cpu().numpy()

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

print("Connecting to DB...")
conn = psycopg2.connect(get_db_url())

# 1. Fetch stored Hypoglycaemia embedding from DB
with conn.cursor() as cur:
    cur.execute("""
        SELECT pt_code, pt_name, pt_embedding::text
        FROM processed.meddra_terms
        WHERE pt_name ILIKE '%hypoglycaem%'
        LIMIT 3
    """)
    rows = cur.fetchall()

if not rows:
    print("ERROR: No Hypoglycaemia PT found in DB!")
    sys.exit(1)

print(f"\nFound {len(rows)} Hypoglycaemia-related PTs:")
for r in rows:
    print(f"  {r[0]}: {r[1]}")

# Parse stored embedding
stored_emb_str = rows[0][2]  # e.g. "[0.123, -0.456, ...]"
stored_emb = np.array([float(x) for x in stored_emb_str.strip("[]").split(",")])
print(f"\nStored embedding: dim={len(stored_emb)}, norm={np.linalg.norm(stored_emb):.4f}")
print(f"  First 5 values: {stored_emb[:5]}")

# 2. Load model locally
print(f"\nLoading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()
print("Model loaded.")

# 3. Encode PT name locally
pt_name = rows[0][1]
local_pt_emb = encode(pt_name, tokenizer, model)
print(f"\nLocal embedding of '{pt_name}':")
print(f"  dim={len(local_pt_emb)}, norm={np.linalg.norm(local_pt_emb):.4f}")
print(f"  First 5 values: {local_pt_emb[:5]}")

# 4. Compare stored vs local PT embedding
sim_self = cosine(stored_emb, local_pt_emb)
print(f"\nCosine similarity (stored vs local PT name encoding): {sim_self:.4f}")
if sim_self > 0.95:
    print("  -> GOOD: embeddings are compatible")
elif sim_self > 0.5:
    print("  -> WARNING: moderate similarity -- possible model version mismatch")
else:
    print("  -> BAD: embeddings are incompatible! Model or encoding mismatch.")

# 5. Encode narrative
narrative = "Patient experienced hypoglycaemia after insulin pump delivered unexpected bolus."
narrative_emb = encode(narrative, tokenizer, model)
sim_narrative = cosine(stored_emb, narrative_emb)
print(f"\nCosine similarity (stored Hypoglycaemia vs narrative): {sim_narrative:.4f}")
if sim_narrative > 0.7:
    print("  -> GOOD: narrative matches Hypoglycaemia well")
elif sim_narrative > 0.4:
    print("  -> OK: moderate match (expected for narrative vs short term)")
else:
    print("  -> LOW: narrative does not match Hypoglycaemia semantically")

# 6. Check pgvector directly
print(f"\nPgvector direct query for '{pt_name}':")
with conn.cursor() as cur:
    cur.execute("SET ivfflat.probes = 100")
    cur.execute("""
        SELECT pt_name, 1 - (pt_embedding <=> %s::vector) AS sim
        FROM processed.meddra_terms
        ORDER BY pt_embedding <=> %s::vector
        LIMIT 5
    """, (str(local_pt_emb.tolist()), str(local_pt_emb.tolist())))
    results = cur.fetchall()

print("  Top 5 results when querying with local 'Hypoglycaemia' embedding:")
for r in results:
    print(f"    {r[1]:.4f}  {r[0]}")

# 7. Direct pgvector test with the actual narrative
print(f"\nPgvector top 10 for the full narrative:")
with conn.cursor() as cur:
    cur.execute("SET ivfflat.probes = 100")
    cur.execute("""
        SELECT pt_name, 1 - (pt_embedding <=> %s::vector) AS sim
        FROM processed.meddra_terms
        ORDER BY pt_embedding <=> %s::vector
        LIMIT 10
    """, (str(narrative_emb.tolist()), str(narrative_emb.tolist())))
    results = cur.fetchall()

for i, r in enumerate(results, 1):
    print(f"  {i}. {r[1]:.4f}  {r[0]}")

# 8. Where is Hypoglycaemia in the full vector ranking?
print(f"\nRank of Hypoglycaemia-related PTs in full vector search:")
with conn.cursor() as cur:
    cur.execute("SET ivfflat.probes = 100")
    cur.execute("""
        WITH ranked AS (
            SELECT pt_name,
                   1 - (pt_embedding <=> %s::vector) AS sim,
                   ROW_NUMBER() OVER (ORDER BY pt_embedding <=> %s::vector) AS rank
            FROM processed.meddra_terms
        )
        SELECT rank, sim, pt_name FROM ranked
        WHERE pt_name ILIKE '%%hypoglycaem%%'
        ORDER BY rank
        LIMIT 5
    """, (str(narrative_emb.tolist()), str(narrative_emb.tolist())))
    results = cur.fetchall()

for r in results:
    print(f"  rank={r[0]}  sim={r[1]:.4f}  {r[2]}")

conn.close()
