"""
Inference-Modul für vigilex.
Lädt das gespeicherte Modell und gibt Recall-Risiko-Scores zurück.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path


def load_model(model_path: str = 'models/lgbm_recall_risk.joblib') -> dict:
    return joblib.load(model_path)


def predict_recall_risk(
    features: pd.DataFrame,
    model_path: str = 'models/lgbm_recall_risk.joblib'
) -> pd.DataFrame:
    """
    Input:  DataFrame mit FEATURE_COLS
    Output: DataFrame mit `recall_prob` und `recall_flag` (0/1)
    """
    artifact  = load_model(model_path)
    model     = artifact['model']
    threshold = artifact['threshold']
    feat_cols = artifact['feature_cols']

    missing = set(feat_cols) - set(features.columns)
    if missing:
        raise ValueError(f'Fehlende Features: {missing}')

    probs = model.predict_proba(features[feat_cols])[:, 1]
    result = features.copy()
    result['recall_prob'] = probs
    result['recall_flag'] = (probs >= threshold).astype(int)
    return result
