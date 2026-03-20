"""
Feature Engineering für vigilex.
Baut Rolling-Window Features und Severity Scores aus MAUDE-Rohdaten.
"""

import pandas as pd
import numpy as np


EVENT_TYPE_SCORE = {
    'D':  4,
    'IN': 3,
    'IL': 3,
    'M':  2,
    'O':  1,
    'N':  1,
}

OUTCOME_SCORE = {
    'D':  5,
    'LT': 4,
    'H':  3,
    'RI': 2,
    'O':  1,
    'N':  1,
    '*':  1,
}

FEATURE_COLS = [
    'complaints_7d', 'complaints_30d', 'complaints_90d',
    'severe_events_7d', 'severe_events_30d', 'severe_events_90d',
    'complaint_accel', 'max_severity', 'mean_severity',
]


def add_severity_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Fügt event_score, outcome_score und severity_score hinzu."""
    df = df.copy()
    df['event_score']   = df['event_type'].map(EVENT_TYPE_SCORE).fillna(1).astype(int)
    df['outcome_score'] = df['patient_outcome'].map(OUTCOME_SCORE).fillna(1).astype(int)
    df['severity_score'] = df[['event_score', 'outcome_score']].max(axis=1)
    return df


def build_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet Rolling-Window Features pro Hersteller + Produktcode.

    Input: MAUDE DataFrame mit Spalten date_of_event, manufacturer_name,
           product_code, report_number, severity_score
    Output: Tages-aggregierter DataFrame mit Rolling Features
    """
    df = df.copy()
    df = df.dropna(subset=['date_of_event', 'manufacturer_name'])
    df = df.sort_values('date_of_event').reset_index(drop=True)

    parts = []
    for (mfr, pc), grp in df.groupby(['manufacturer_name', 'product_code'], sort=False):
        grp = grp.set_index('date_of_event').sort_index()

        daily = grp.resample('D').agg(
            complaints=('report_number', 'count'),
            severe_events=('severity_score', lambda x: (x >= 3).sum()),
            max_severity=('severity_score', 'max'),
            mean_severity=('severity_score', 'mean'),
        )

        for window, label in [(7, '7d'), (30, '30d'), (90, '90d')]:
            daily[f'complaints_{label}']    = daily['complaints'].rolling(f'{window}D', min_periods=1).sum()
            daily[f'severe_events_{label}'] = daily['severe_events'].rolling(f'{window}D', min_periods=1).sum()

        daily['complaint_accel'] = daily['complaints_30d'] / (daily['complaints_90d'] + 1e-9)
        daily['manufacturer_name'] = mfr
        daily['product_code']      = pc
        parts.append(daily.reset_index())

    return pd.concat(parts, ignore_index=True)


def label_recall_risk(
    df_feat: pd.DataFrame,
    recalls: pd.DataFrame,
    horizon_days: int = 365
) -> pd.DataFrame:
    """
    Fügt recall_label (0/1) und recall_class hinzu.

    recalls muss Spalten haben: recall_date, recalling_firm_norm, classification
    """
    df_feat = df_feat.copy()
    df_feat['recall_label'] = 0
    df_feat['recall_class'] = 'None'
    df_feat['_mfr_norm'] = df_feat['manufacturer_name'].str.upper().str.strip()

    for idx, row in df_feat.iterrows():
        horizon_end = row['date_of_event'] + pd.Timedelta(days=horizon_days)
        matches = recalls[
            recalls['recalling_firm_norm'].str.contains(
                row['_mfr_norm'][:10], na=False, regex=False
            ) &
            (recalls['recall_date'] >= row['date_of_event']) &
            (recalls['recall_date'] <= horizon_end)
        ]
        if len(matches) > 0:
            df_feat.at[idx, 'recall_label'] = 1
            df_feat.at[idx, 'recall_class'] = matches['classification'].iloc[0]

    return df_feat.drop(columns=['_mfr_norm'])
