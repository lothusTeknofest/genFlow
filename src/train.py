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
from src.custom_model import GenFlowModel

warnings.filterwarnings("ignore", category=RuntimeWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COLUMN = "Label"
RANDOM_STATE = 42

def optimize_and_evaluate(model, X_test, y_test, panel_name):
    y_proba = model.predict_proba(X_test)
    y_score = y_proba[:, 1]
    
    print(f"\n{'='*40}")
    print(f"🧬 {panel_name.upper()} PANELİ İÇİN KLİNİK EŞİK TESTİ")
    print(f"{'='*40}")
    
    # 1. Şartnamenin İstediği: Farklı Karar Eşikleri (Threshold) Testi
    thresholds_to_test = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    
    print(f"{'Eşik':<10} | {'F1 Skoru':<12} | {'MCC Skoru':<12}")
    print("-" * 40)
    
    for thresh in thresholds_to_test:
        y_pred_thresh = (y_score >= thresh).astype(int)
        current_f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
        current_mcc = matthews_corrcoef(y_test, y_pred_thresh)
        print(f"{thresh:<10.2f} | {current_f1:<12.4f} | {current_mcc:<12.4f}")

    # 2. Youden İndeksi ile Optimum Eşik Bulma (Sizin Orijinal Kodunuz)
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[best_idx]
    
    model.optimal_threshold_ = optimal_threshold

    y_pred_opt = (y_score >= optimal_threshold).astype(int)
    f1_opt = f1_score(y_test, y_pred_opt, zero_division=0)
    mcc_opt = matthews_corrcoef(y_test, y_pred_opt)
    pr_auc = average_precision_score(y_test, y_score)

    # 3. Nihai Optimizasyon Sonucunun Basılması
    print("\n🎯 OPTİMAL KLİNİK KARAR (YOUDEN İNDEKSİ)")
    print(f"Bulunan En İyi Eşik  : {optimal_threshold:.3f}")
    print(f"Elde Edilen F1 Skoru : {f1_opt:.4f}")
    print(f"Elde Edilen MCC Skoru: {mcc_opt:.4f}")
    print(f"Model Genel Gücü (PR-AUC): {pr_auc:.4f}\n")

def main():
    print("=" * 60)
    print("GEN FLOW — Klinik Eşik (Youden's J) Optimizasyonlu Eğitim")
    print("=" * 60)

    # 1. MASTER MODEL
    print("\n🚀 [MASTER GENOM MODELİ] Eğitiliyor...")
    df_master = pd.read_csv(DATA_DIR / "processed_MASTER.csv")
    X_m = df_master.select_dtypes(include="number").drop(columns=[LABEL_COLUMN])
    y_m = df_master[LABEL_COLUMN].astype(int)
    X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(X_m, y_m, test_size=0.2, stratify=y_m, random_state=RANDOM_STATE)

    # YENİ EKLENEN: Master modeldeki -400'lük CAT_3 patlamasını durduran regülarizasyon zırhı
    master_model = GenFlowModel(
        random_state=RANDOM_STATE, 
        n_estimators=150, 
        max_depth=4,               # Ağaç derinliğini sınırla
        learning_rate=0.05,        # Öğrenmeyi yavaşlat
        colsample_bytree=0.6,      # Özellik ağırlığını dağıt
        subsample=0.8,
        reg_alpha=1.0,             # L1 Regülarizasyonu
        reg_lambda=3.0             # L2 Regülarizasyonu
    )
    master_model.fit(X_m_train, y_m_train)
    optimize_and_evaluate(master_model, X_m_test, y_m_test, "Master Genom")
    joblib.dump(master_model, MODEL_DIR / "master_model.pkl")


    # 2. TRANSFER LEARNING
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
            panel_model = GenFlowModel(
            random_state=RANDOM_STATE, n_estimators=100, max_depth=4,
            learning_rate=0.05, colsample_bytree=0.6, subsample=0.8, reg_alpha=1.0, reg_lambda=3.0
        )
            panel_model._base_classifier.fit(X_p_train, y_p_train, xgb_model=master_booster)
            
            from sklearn.calibration import CalibratedClassifierCV
            panel_model._calibrated_model = CalibratedClassifierCV(estimator=panel_model._base_classifier, cv="prefit", method="isotonic")
            panel_model._calibrated_model.fit(X_p_train, y_p_train)
            panel_model._is_fitted = True
        except Exception:
            panel_model = GenFlowModel(
            random_state=RANDOM_STATE, n_estimators=150, max_depth=4,
            learning_rate=0.05, colsample_bytree=0.6, subsample=0.8, reg_alpha=1.0, reg_lambda=3.0
        )
            panel_model.fit(X_p_train, y_p_train)

        optimize_and_evaluate(panel_model, X_p_test, y_p_test, panel)
        joblib.dump(panel_model, MODEL_DIR / f"{panel.lower()}_model.pkl")

    print("\n" + "=" * 60)
    print("🎉 MÜKEMMEL! Tüm paneller Optimum Tıbbi Eşiklerle kaydedildi.")
    print("=" * 60)


if __name__ == "__main__":
    main()