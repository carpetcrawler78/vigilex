"""
Training-Pipeline für vigilex Recall Risk Modell.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

import lightgbm as lgb
import optuna
from sklearn.metrics import average_precision_score, roc_auc_score, fbeta_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURE_COLS = [
    'complaints_7d', 'complaints_30d', 'complaints_90d',
    'severe_events_7d', 'severe_events_30d', 'severe_events_90d',
    'complaint_accel', 'max_severity', 'mean_severity',
]
TARGET = 'recall_label'


def temporal_split(df: pd.DataFrame):
    """Zeitlicher Split: Train < 2023, Val 2023, Test >= 2024."""
    df['date_of_event'] = pd.to_datetime(df['date_of_event'])
    train = df[df['date_of_event'] < '2023-01-01']
    val   = df[(df['date_of_event'] >= '2023-01-01') & (df['date_of_event'] < '2024-01-01')]
    test  = df[df['date_of_event'] >= '2024-01-01']
    return train, val, test


def tune_lightgbm(
    X_train, y_train,
    X_val, y_val,
    n_trials: int = 50
) -> dict:
    """Optuna Hyperparameter-Suche für LightGBM. Optimiert PR-AUC."""
    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    def objective(trial):
        params = {
            'objective':         'binary',
            'metric':            'average_precision',
            'verbosity':         -1,
            'scale_pos_weight':  scale_pos,
            'n_estimators':      trial.suggest_int('n_estimators', 100, 800),
            'learning_rate':     trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
            'num_leaves':        trial.suggest_int('num_leaves', 16, 128),
            'max_depth':         trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha':         trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda':        trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'random_state':      42,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        return average_precision_score(y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params


def find_best_threshold(model, X_val, y_val, beta: float = 2.0) -> float:
    """Findet Threshold der den F-beta Score maximiert."""
    y_prob = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.1, 0.9, 0.02)
    scores = [fbeta_score(y_val, (y_prob >= t).astype(int), beta=beta) for t in thresholds]
    return float(thresholds[np.argmax(scores)])


def train_and_save(
    features_path: str = 'data/processed/features_labeled.parquet',
    output_path: str   = 'models/lgbm_recall_risk.joblib',
    n_trials: int      = 50,
):
    """Kompletter Training-Lauf: laden → tunen → evaluieren → speichern."""
    df = pd.read_parquet(features_path)
    train_df, val_df, test_df = temporal_split(df)

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET]
    X_val,   y_val   = val_df[FEATURE_COLS],   val_df[TARGET]
    X_test,  y_test  = test_df[FEATURE_COLS],  test_df[TARGET]

    print(f'Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}')

    best_params = tune_lightgbm(X_train, y_train, X_val, y_val, n_trials=n_trials)
    scale_pos   = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    best_params.update({'objective': 'binary', 'verbosity': -1,
                        'scale_pos_weight': scale_pos, 'random_state': 42})

    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False)])

    threshold = find_best_threshold(model, X_val, y_val)
    y_prob    = model.predict_proba(X_test)[:, 1]
    y_pred    = (y_prob >= threshold).astype(int)

    print(f'\n=== Test-Set ===')
    print(f'ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}')
    print(f'PR-AUC:  {average_precision_score(y_test, y_prob):.4f}')
    print(f'F2:      {fbeta_score(y_test, y_pred, beta=2):.4f}')

    artifact = {
        'model':        model,
        'threshold':    threshold,
        'feature_cols': FEATURE_COLS,
        'best_params':  best_params,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    print(f'Modell gespeichert: {output_path}')
    return artifact


if __name__ == '__main__':
    train_and_save()
