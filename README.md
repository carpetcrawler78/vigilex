# vigilex
**Adverse Event Signal Detection for Medical Devices**

ML pipeline for classifying problem types and predicting recall risk from FDA MAUDE adverse event reports. Built as part of the neue fische AI Engineering Bootcamp 2025/2026.

---

## What it does

Medical device manufacturers and regulators receive thousands of adverse event reports every month. vigilex uses machine learning to:

- **Classify problem types** (software failure, hardware defect, user error, design flaw) from structured report fields
- **Predict recall risk** – flags devices with elevated recall probability based on complaint patterns over time
- **Explain findings in regulatory context** (Phase II) – a RAG agent maps ML outputs to relevant MDR and EU AI Act articles

---

## Data Sources

| Source | Usage |
|---|---|
| [FDA MAUDE](https://open.fda.gov/apis/device/event/) – ~15M adverse event reports | ML training data |
| [FDA Recall Database](https://open.fda.gov/apis/device/recall/) | Label generation (recall yes/no) |
| EU MDR 2017/745, EU AI Act | RAG knowledge base (Phase II) |
| BfArM Field Safety Notices | RAG knowledge base (Phase II) |

---

## Project Structure

```
vigilex/
├── notebooks/
│   ├── 01_openfda_eda.ipynb        # API access, data pull, EDA
│   ├── 02_feature_engineering.ipynb # Rolling windows, severity scores
│   ├── 03_modeling.ipynb            # LightGBM, Optuna, evaluation
│   └── 04_rag_agent.ipynb           # Phase II: LangChain + FAISS
├── src/
│   ├── data/                        # API client, data loading
│   ├── features/                    # Feature builders
│   └── models/                      # Training, evaluation
├── models/                          # Saved model artifacts
├── data/                            # Not tracked (see .gitignore)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # macOS/Linux

pip install -r requirements.txt
jupyter lab
```

Optional: Get a free openFDA API key at https://open.fda.gov/apis/authentication/ (raises rate limit from 1k to 120k requests/day).

Add it to a `.env` file:
```
OPENFDA_API_KEY=your_key_here
```

---

## Roadmap

**Phase I – ML Pipeline (Weeks 1–4)**
- [x] openFDA API client + EDA
- [x] Feature engineering: rolling complaint windows, severity scoring
- [x] Recall label join (MAUDE × FDA Recall DB)
- [x] Baseline models: Logistic Regression, Random Forest
- [x] LightGBM + Optuna tuning
- [ ] Streamlit demo: device type → risk score + feature importance

**Phase II – AI Engineering (Weeks 5–8)**
- [ ] RAG knowledge base: MDR, EU AI Act, BfArM FSNs
- [ ] LangChain + FAISS retrieval pipeline
- [ ] Regulatory explanation agent
- [ ] End-to-end demo: complaint input → ML score → regulatory context

---

## Background

Built by Thomas Heger – Dr. sc. ETH Biochemistry, former Clinical Data Manager at DKFZ (German Cancer Research Center), Heidelberg. Domain expertise in post-market surveillance (REQUITE, RADprecise studies), GCP documentation, and EU medical device regulation informs both the feature engineering and regulatory framing of this project.

---

## License

MIT
