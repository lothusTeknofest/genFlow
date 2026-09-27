"""
GEN FLOW — Synonymous Mutation Modeling
Evaluation Module to compute final metrics, plot threshold optimizations, and run SHAP analysis.
"""

from __future__ import annotations

import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, average_precision_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Resolve pickle import issues
from src.model_trainer import SupervisedSynonymousModel
sys.modules['__main__'].SupervisedSynonymousModel = SupervisedSynonymousModel
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

FEATURE_COLUMNS = [
    'RSCU_WT', 'RSCU_MUT', 'Delta_RSCU',
    'CAI_WT', 'CAI_MUT', 'Delta_CAI',
    'wt_MFE', 'mut_MFE', 'Delta_Delta_G',
    'phyloP', 'phastCons', 'splicing_dist_to_junction'
]
LABEL_COLUMN = 'Label'


def main():
    model_path = MODEL_DIR / "supervised_synonymous_model.pkl"
    data_path = DATA_DIR / "processed_clinvar_synonymous.csv"
    
    if not model_path.exists() or not data_path.exists():
        print("Error: Model or data not found. Please run feature_extractor.py and model_trainer.py first.")
        sys.exit(1)
        
    # 1. Load model and data
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0.0)
    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN]
    
    # 2. Predict and evaluate
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y, y_pred)
    pr_auc = average_precision_score(y, y_proba)
    
    print("=" * 60)
    print("=== GEN FLOW - FINAL EVALUATION RESULTS (SUPERVISED MODEL) ===")
    print("=" * 60)
    print(f"Total Test Variants  : {len(df)}")
    print(f"Pathogenic Variants  : {y.sum()}")
    print(f"Benign Variants      : {len(df) - y.sum()}")
    print(f"Optimal Threshold    : {model.optimal_threshold_:.3f}")
    print("-" * 60)
    print(f"Precision            : {prec:.4f}")
    print(f"Recall (Sensitivity) : {rec:.4f}")
    print(f"F1 Score             : {f1:.4f}")
    print(f"MCC Score            : {mcc:.4f}")
    print(f"PR-AUC               : {pr_auc:.4f}")
    print("=" * 60)
    
    # 3. Plot Youden threshold optimization curve
    thresholds = np.linspace(0.0, 1.0, 100)
    f1_scores = []
    for t in thresholds:
        y_t = (y_proba >= t).astype(int)
        f1_scores.append(f1_score(y, y_t, zero_division=0))
        
    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, f1_scores, color='#2ca02c', lw=3, label='F1 Score Change')
    plt.axvline(x=model.optimal_threshold_, color='red', linestyle='--', lw=2,
                label=f'Optimal Youden Threshold = {model.optimal_threshold_:.3f}')
    plt.title('Karar Eşiği (Threshold) ve F1 Skoru İlişkisi', fontweight='bold', fontsize=14)
    plt.xlabel('Karar Eşiği (Threshold)', fontweight='bold')
    plt.ylabel('Başarım (F1 Skoru)', fontweight='bold')
    plt.legend(loc='lower center')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    youden_plot_path = PROJECT_ROOT / "Youden_Threshold_Optimization.png"
    plt.savefig(youden_plot_path, dpi=300)
    print(f"Saved Youden optimization plot to {youden_plot_path}")
    plt.close()
    
    # 4. SHAP Analysis
    print("Running SHAP feature importance analysis...")
    X_scaled = model.scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS)
    
    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(X_scaled_df)
    
    # Standardize SHAP output structure depending on xgboost/shap versions
    if isinstance(shap_values, list) and len(shap_values) == 2:
        # For classification models, SHAP might return a list of arrays (one per class)
        shap_values = shap_values[1]
        
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_scaled_df, show=False)
    plt.title('SHAP Özellik Önem Analizi (Sessiz Mutasyon Modellemesi)', fontweight='bold', fontsize=14, pad=15)
    plt.tight_layout()
    
    shap_plot_path = PROJECT_ROOT / "SHAP_Feature_Importance.png"
    plt.savefig(shap_plot_path, dpi=300)
    print(f"Saved SHAP plot to {shap_plot_path}")
    plt.close()


if __name__ == "__main__":
    main()
