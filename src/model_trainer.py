"""
GEN FLOW — Synonymous Mutation Modeling
Modeling Module to train a supervised XGBoost classifier on synonymous mutations.
"""

from __future__ import annotations

import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, roc_curve, average_precision_score
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
    'RSCU_WT', 'RSCU_MUT', 'Delta_RSCU',
    'CAI_WT', 'CAI_MUT', 'Delta_CAI',
    'wt_MFE', 'mut_MFE', 'Delta_Delta_G',
    'phyloP', 'phastCons', 'splicing_dist_to_junction'
]
LABEL_COLUMN = 'Label'
RANDOM_STATE = 42


class SupervisedSynonymousModel:
    def __init__(self, random_state: int = RANDOM_STATE):
        self.scaler = RobustScaler()
        self.model = XGBClassifier(
            max_depth=4,
            learning_rate=0.05,
            n_estimators=150,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric="logloss"
        )
        self.optimal_threshold_ = 0.5

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        # Calculate optimal threshold using training predictions
        y_proba = self.model.predict_proba(X_scaled)[:, 1]
        fpr, tpr, thresholds = roc_curve(y, y_proba)
        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        self.optimal_threshold_ = thresholds[best_idx]
        print(f"Optimal Youden threshold determined: {self.optimal_threshold_:.3f}")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        y_proba = self.predict_proba(X)[:, 1]
        return (y_proba >= self.optimal_threshold_).astype(int)


def train_and_validate():
    input_path = DATA_DIR / "processed_clinvar_synonymous.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Processed file not found: {input_path}")
        
    df = pd.read_csv(input_path)
    
    # Preprocess missing values if any
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0.0)
    
    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN]
    groups = df['Gene']
    
    print("=" * 60)
    print("GEN FLOW — Supervised Synonymous Mutation modeling Cross Validation")
    print("=" * 60)
    print(f"Dataset Size: {len(df)} variants")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Groups (Genes): {groups.unique()}")
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        val_genes = groups.iloc[val_idx].unique()
        
        # Fit model on this fold
        fold_model = SupervisedSynonymousModel()
        fold_model.fit(X_train, y_train)
        
        # Predict on validation set
        y_score = fold_model.predict_proba(X_val)[:, 1]
        y_pred = fold_model.predict(X_val)
        
        # Compute metrics
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_val, y_pred)
        pr_auc = average_precision_score(y_val, y_score)
        
        print(f"\nFold {fold+1} Validation (Held-out Genes: {val_genes})")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1 Score : {f1:.4f}")
        print(f"  MCC      : {mcc:.4f}")
        print(f"  PR-AUC   : {pr_auc:.4f}")
        
        metrics.append({
            'fold': fold+1,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'mcc': mcc,
            'pr_auc': pr_auc
        })
        
    df_metrics = pd.DataFrame(metrics)
    print("\n" + "=" * 60)
    print("=== CROSS-VALIDATION SUMMARY METRICS ===")
    print("=" * 60)
    print(f"Mean Precision: {df_metrics['precision'].mean():.4f}")
    print(f"Mean Recall   : {df_metrics['recall'].mean():.4f}")
    print(f"Mean F1 Score : {df_metrics['f1'].mean():.4f}")
    print(f"Mean MCC      : {df_metrics['mcc'].mean():.4f}")
    print(f"Mean PR-AUC   : {df_metrics['pr_auc'].mean():.4f}\n")
    
    # Train final model on entire dataset
    print("Training final model on full dataset...")
    final_model = SupervisedSynonymousModel()
    final_model.fit(X, y)
    
    model_path = MODEL_DIR / "supervised_synonymous_model.pkl"
    joblib.dump(final_model, model_path)
    print(f"Saved final model to {model_path}")


if __name__ == "__main__":
    train_and_validate()
