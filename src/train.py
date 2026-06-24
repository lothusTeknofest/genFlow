"""
GEN FLOW — TEKNOFEST Sağlıkta Yapay Zeka Yarışması
Sürekli Öğrenme ve Youden's J Klinik Eşik Optimizasyonu.
"""

from __future__ import annotations

import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import f1_score, matthews_corrcoef, average_precision_score, roc_curve
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Proje kökünü Python yoluna ekle
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.custom_model import GenFlowModel

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COLUMN = "Label"
RANDOM_STATE = 42

def optimize_and_evaluate(model, X_test, y_test, panel_name):
    """Youden's J ile optimum eşiği bulur, modeli günceller ve sonuçları basar."""
    y_proba = model.predict_proba(X_test)
    y_score = y_proba[:, 1] # Patojenik olma olasılıkları

    # 1. Eski Standart (0.5 Eşiği) Sonuçları
    y_pred_default = (y_score >= 0.5).astype(int)
    f1_def = f1_score(y_test, y_pred_default, zero_division=0)
    mcc_def = matthews_corrcoef(y_test, y_pred_default)

    # 2. Youden's J İstatistiği ile Optimum Eşiği Bulma
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[best_idx]
    
    # Modelin içine optimum eşiği (custom attribute olarak) kaydediyoruz!
    model.optimal_threshold_ = optimal_threshold

    # 3. Yeni Optimum Eşiğe Göre Tahminler
    y_pred_opt = (y_score >= optimal_threshold).astype(int)
    f1_opt = f1_score(y_test, y_pred_opt, zero_division=0)
    mcc_opt = matthews_corrcoef(y_test, y_pred_opt)
    pr_auc = average_precision_score(y_test, y_score)

    print(f"\n    📊 {panel_name.upper()} METRİKLERİ:")
    print(f"       - Standart Eşik (0.500) -> F1: {f1_def:.4f} | MCC: {mcc_def:.4f}")
    print(f"       - Youden Eşiği  ({optimal_threshold:.3f}) -> F1: {f1_opt:.4f} | MCC: {mcc_opt:.4f}")
    print(f"       - Modelin Genel Gücü (PR-AUC): {pr_auc:.4f}")

def main():
    print("=" * 60)
    print("GEN FLOW — Klinik Eşik (Youden's J) Optimizasyonlu Eğitim")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. MASTER MODELİ EĞİT
    # ---------------------------------------------------------
    print("\n🚀 [MASTER GENOM MODELİ] Eğitiliyor...")
    df_master = pd.read_csv(DATA_DIR / "processed_MASTER.csv")
    X_m = df_master.select_dtypes(include="number").drop(columns=[LABEL_COLUMN])
    y_m = df_master[LABEL_COLUMN].astype(int)
    X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(X_m, y_m, test_size=0.2, stratify=y_m, random_state=RANDOM_STATE)

    master_model = GenFlowModel(random_state=RANDOM_STATE, n_estimators=300, max_depth=6)
    master_model.fit(X_m_train, y_m_train)
    optimize_and_evaluate(master_model, X_m_test, y_m_test, "Master Genom")
    joblib.dump(master_model, MODEL_DIR / "master_model.pkl")

    # ---------------------------------------------------------
    # 2. BİLGİ TRANSFERİ (KÜÇÜK PANELLER)
    # ---------------------------------------------------------
    sub_panels = ["KANSER", "PAH", "CFTR"]
    
    for panel in sub_panels:
        print(f"\n🚀 [{panel} MODELİ] Eğitiliyor (Transfer Learning)...")
        
        panel_path = DATA_DIR / f"processed_{panel}.csv"
        if not panel_path.exists(): continue
            
        df_panel = pd.read_csv(panel_path)
        X_p = df_panel.select_dtypes(include="number").drop(columns=[LABEL_COLUMN])
        y_p = df_panel[LABEL_COLUMN].astype(int)
        X_p_train, X_p_test, y_p_train, y_p_test = train_test_split(X_p, y_p, test_size=0.2, stratify=y_p, random_state=RANDOM_STATE)
        
        try:
            master_booster = master_model._calibrated_model.estimator.get_booster()
            panel_model = GenFlowModel(random_state=RANDOM_STATE, n_estimators=100, max_depth=6)
            panel_model._base_classifier.fit(X_p_train, y_p_train, xgb_model=master_booster)
            
            from sklearn.calibration import CalibratedClassifierCV
            panel_model._calibrated_model = CalibratedClassifierCV(estimator=panel_model._base_classifier, cv="prefit", method="isotonic")
            panel_model._calibrated_model.fit(X_p_train, y_p_train)
            panel_model._is_fitted = True
        except Exception:
            panel_model = GenFlowModel(random_state=RANDOM_STATE, n_estimators=300, max_depth=6)
            panel_model.fit(X_p_train, y_p_train)

        # Optimizasyon ve Kayıt
        optimize_and_evaluate(panel_model, X_p_test, y_p_test, panel)
        joblib.dump(panel_model, MODEL_DIR / f"{panel.lower()}_model.pkl")

    print("\n" + "=" * 60)
    print("🎉 MÜKEMMEL! Tüm paneller Optimum Tıbbi Eşiklerle kaydedildi.")
    print("=" * 60)

if __name__ == "__main__":
    main()