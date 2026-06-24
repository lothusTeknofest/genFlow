"""
GEN FLOW — TEKNOFEST Sağlıkta Yapay Zeka Yarışması
Kapsamlı Dashboard: SHAP, Toplu Analiz, What-If ve Hibrit (Missense+Synonymous) Modülleri.
"""

from __future__ import annotations

import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import shap
import matplotlib.pyplot as plt
import io

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Veri ve Model Yükleme
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(panel_name: str):
    model_path = PROJECT_ROOT / "models" / f"{panel_name.lower()}_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"⚠️ {panel_name} modeli bulunamadı!")
    return joblib.load(model_path)

@st.cache_data
def load_dataset(panel_name: str) -> pd.DataFrame:
    csv_path = PROJECT_ROOT / "data" / f"processed_{panel_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"⚠️ {panel_name} verisi bulunamadı!")
    return pd.read_csv(csv_path)

@st.cache_data
def load_feature_template(panel_name: str) -> pd.Series:
    df = load_dataset(panel_name)
    numeric = df.select_dtypes(include="number").drop(columns=["Label"], errors="ignore")
    return numeric.median()

# ---------------------------------------------------------------------------
# Arayüz
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="GEN FLOW - Klinik Karar Destek", page_icon="🧬", layout="wide")

    st.title("GEN FLOW — Akıllı Varyant Patojenite Analizi")
    st.markdown("TEKNOFEST Sağlıkta Yapay Zeka · **Sessiz Mutasyonlar Takımı (XAI Destekli)**")
    
    # --- SOL MENÜ ---
    st.sidebar.header("⚙️ Klinik Panel Ayarları")
    st.sidebar.caption("Not: Yarışma ana odağı gereği şu an Missense panelleri aktiftir.")
    selected_panel = st.sidebar.selectbox("Hastalık Paneli Seçin:", ("KANSER", "PAH", "CFTR"))
    st.sidebar.divider()
    
    try:
        model = load_model(selected_panel)
        df_full = load_dataset(selected_panel)
        X_full = df_full.select_dtypes(include="number").drop(columns=["Label"], errors="ignore")
        feature_template = load_feature_template(selected_panel)
        threshold = getattr(model, "optimal_threshold_", 0.5)
        
        st.sidebar.success(f"✅ {selected_panel} (Missense) Zekası Aktif")
        st.sidebar.metric(label="Klinik Karar Eşiği (Youden)", value=f"{threshold:.3f}")
        st.sidebar.caption(f"Risk %{threshold*100:.1f} üzerindeyse PATOJENİK kabul edilir.")
    except Exception as e:
        st.error(str(e))
        return

    # Hafıza Yönetimi
    if "rand_patient" not in st.session_state or st.session_state.get("last_p") != selected_panel:
        st.session_state.rand_patient = X_full.iloc[[0]].copy()
        st.session_state.last_p = selected_panel

    # --- 5 HARİKA SEKME (İSTENİLEN SIRALAMAYLA) ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔬 Gerçek Hasta Simülasyonu", 
        "📂 Toplu CSV Analizi",
        "✍️ Özel Senaryo (What-If)", 
        "🧬 Salt Sessiz (Synonymous) Modülü",
        "🔄 Kademeli Hibrit (Missense + Synonymous)"
    ])

    # ---------------------------------------------------------
    # SEKME 1: GERÇEK HASTA SİMÜLASYONU VE SHAP
    # ---------------------------------------------------------
    with tab1:
        st.subheader("Gerçek Hasta Karar Simülasyonu ve XAI (Açıklanabilirlik)")
        st.write("Sistemin nasıl karar verdiğini jüriye göstermek için veri setinden rastgele bir kayıt çekilir.")
        
        if st.button("🎲 Veri Setinden Rastgele Bir Hasta Getir", type="primary", key="btn_rand"):
            random_idx = np.random.randint(0, len(X_full))
            st.session_state.rand_patient = X_full.iloc[[random_idx]].copy()
            
        patient_data = st.session_state.rand_patient
        proba = model.predict_proba(patient_data)
        risk_score = proba[0, 1]
        is_pathogenic = risk_score >= threshold
        
        st.divider()
        colA, colB = st.columns([1, 2])
        
        with colA:
            st.markdown("### Yapay Zeka Kararı")
            if is_pathogenic:
                st.error(f"⚠️ **PATOJENİK VARYANT**\n\nHesaplanan Risk: **%{risk_score*100:.1f}**\n\n(Eşik: %{threshold*100:.1f})")
            else:
                st.success(f"✅ **BENIGN (SAĞLIKLI) VARYANT**\n\nHesaplanan Risk: **%{risk_score*100:.1f}**\n\n(Eşik: %{threshold*100:.1f})")
        
        with colB:
            st.markdown("### Neden Bu Kararı Verdim? (SHAP Analizi)")
            try:
                calibrated_clf = model._calibrated_model.calibrated_classifiers_[0]
                xgb_core = getattr(calibrated_clf, 'estimator', getattr(calibrated_clf, 'base_estimator', None))
                if xgb_core is None: xgb_core = model._base_classifier
                    
                explainer = shap.TreeExplainer(xgb_core)
                shap_values = explainer(patient_data)
                
                fig, ax = plt.subplots(figsize=(6, 4))
                shap.plots.waterfall(shap_values[0], max_display=7, show=False)
                st.pyplot(fig)
                plt.clf() 
            except Exception as e:
                st.warning(f"Grafik çizilemedi: {str(e)}")

    # ---------------------------------------------------------
    # SEKME 2: TOPLU CSV ANALİZİ (LABORATUVAR)
    # ---------------------------------------------------------
    with tab2:
        st.subheader(f"Toplu {selected_panel} Varyant Listesi Yükle")
        uploaded_file = st.file_uploader("İşlenmiş CSV Dosyası Seçin", type=["csv"], key="bulk_upload")
        
        if uploaded_file is not None:
            bulk_df = pd.read_csv(uploaded_file)
            if st.button("🚀 Toplu Analizi Başlat", type="primary"):
                X_bulk = bulk_df.select_dtypes(include="number").drop(columns=["Label"], errors="ignore")
                for col in X_full.columns:
                    if col not in X_bulk.columns: X_bulk[col] = X_full[col].median()
                X_bulk = X_bulk[X_full.columns]
                
                bulk_scores = model.predict_proba(X_bulk)[:, 1]
                bulk_preds = ["PATOJENİK" if s >= threshold else "BENIGN" for s in bulk_scores]
                
                result_df = bulk_df.copy()
                result_df["Tahmin"] = bulk_preds
                result_df["Patojenik_Risk_%"] = (bulk_scores * 100).round(2)
                
                st.dataframe(result_df[["Variant_ID", "Tahmin", "Patojenik_Risk_%"] if "Variant_ID" in result_df.columns else result_df.columns])

    # ---------------------------------------------------------
    # SEKME 3: ÖZEL SENARYO (WHAT-IF) SİMÜLASYONU
    # ---------------------------------------------------------
    with tab3:
        st.subheader("✍️ Değerleri Kendiniz Belirleyin")
        st.write("Aşağıdaki kritik genetik değerleri el ile değiştirerek yapay zekanın risk oranını nasıl güncellediğini canlı olarak test edin.")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        manual_inputs = {}
        
        with col1: manual_inputs["AL_1"] = st.number_input("AL_1", value=0.000001, format="%.6f", key="m_al1")
        with col2: manual_inputs["AL_7"] = st.number_input("AL_7", value=0.000000, format="%.6f", key="m_al7")
        with col3: manual_inputs["EK_3"] = st.number_input("EK_3", value=0.500000, format="%.6f", key="m_ek3")
        with col4: manual_inputs["EK_4"] = st.number_input("EK_4", value=0.500000, format="%.6f", key="m_ek4")
        with col5: manual_inputs["EK_9"] = st.number_input("EK_9", value=0.500000, format="%.6f", key="m_ek9")

        custom_patient = pd.DataFrame([feature_template])
        for feat, val in manual_inputs.items():
            if feat in custom_patient.columns: custom_patient[feat] = val
                
        m_proba = model.predict_proba(custom_patient)
        m_risk_score = m_proba[0, 1]
        
        st.divider()
        if m_risk_score >= threshold:
            st.error(f"⚠️ **SONUÇ: PATOJENİK** (Risk Oranı: %{m_risk_score*100:.1f} >= Eşik: %{threshold*100:.1f})")
        else:
            st.success(f"✅ **SONUÇ: BENIGN** (Risk Oranı: %{m_risk_score*100:.1f} < Eşik: %{threshold*100:.1f})")

    # ---------------------------------------------------------
    # SEKME 4: SESSİZ MUTASYON (SADECE SYNONYMOUS)
    # ---------------------------------------------------------
    with tab4:
        st.subheader("🧬 Özel Modül: Salt Sessiz (Synonymous) Mutasyonlar")
        st.info("Bu alan, takımımızın ismini taşıyan 'Sessiz Mutasyonlar' projemizin veri setleri için ayrılmıştır.")
        st.write("Veriler sisteme entegre edildiğinde, amino asit dizilimini değiştirmemesine rağmen **RNA Splicing** ve **Codon Usage Bias** gibi mekanizmalarla hastalığa sebep olan gizli varyantlar bu modül üzerinden incelenecektir.")
        st.button("Sessiz Mutasyon Analizini Başlat (Veri Bekleniyor)", disabled=True, key="syn_btn")

    # ---------------------------------------------------------
    # SEKME 5: HİBRİT KADEMELİ GÜVENLİK SİSTEMİ (JÜRİ VURUCUSU)
    # ---------------------------------------------------------
    with tab5:
        st.subheader("🔄 Kademeli Taramalı Hibrit Sistem (Missense + Synonymous)")
        st.markdown("Yarışma standartlarında varyantlar Missense modelinden geçip 'Benign' (Zararsız) etiketi aldığında işlem biter. **Ancak bizim projemizde bitmez!**")
        
        st.warning("**GEN FLOW Hibrit Çalışma Mantığı:**")
        st.markdown("""
        1. **Birinci Aşama:** Hastanın verisi ana yarışma paneline (Missense) girer. 
           * Eğer **Patojenik** çıkarsa -> Süreç durur, hasta tedaviye alınır.
           * Eğer **Benign (Sağlıklı)** çıkarsa -> Süreç **DURMAZ**, veri 2. aşamaya gönderilir.
        2. **İkinci Aşama (Güvenlik Ağı):** Missense'in 'zararsız' dediği bu varyantlar, ekibimizin geliştirdiği **Sessiz Mutasyon (Synonymous)** ağına takılır. Amino asit değişmese bile RNA düzeyindeki gizli tehditler burada aranır.
        3. **Nihai Karar:** Ancak iki yapay zeka modelinden de "Benign" onayı alan varyantlar gerçekten sağlıklı kabul edilir.
        """)
        
        st.info("Klinik Yanlış Negatif (False Negative) oranını sıfıra indirmeyi hedefleyen bu Çift Kademeli Mimari, veriler yüklendiğinde bu sekmeden otomatik tarama yapacaktır.")
        st.button("Hibrit Kademeli Taramayı Başlat (Sistem Hazır, Veri Bekleniyor)", disabled=True, key="hybrid_btn")

if __name__ == "__main__":
    main()