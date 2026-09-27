<div align="center">

# 🧬 GEN FLOW
### Missense ve Synonymous Varyantların Hibrit Patojenite Analizi
*TEKNOFEST Sağlıkta Yapay Zeka Yarışması Projesi*

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Optimized-brightgreen.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-orange.svg)](https://shap.readthedocs.io/)

</div>

---

## 📌 Proje Hakkında

**GEN FLOW**, klinik genomik araştırmalarda genetik varyantların (mutasyonların) hastalığa yol açma potansiyellerini (patojenite) yüksek doğrulukla tahmin eden, hibrit ve yapay zeka destekli bir **Klinik Karar Destek Sistemidir (CDSS)**. 

Mevcut modeller (AlphaMissense vb.) genellikle sadece amino asit değişimlerine (*missense*) odaklanırken; bu proje genomdaki en büyük kör nokta olan **Synonymous (Sessiz) Mutasyonların** yıkıcı etkilerini de analiz eder. Sistem, RNA kırpılma (splicing) bozukluklarını, Kodon Yanlılığını (RSCU/CAI) ve mRNA katlanma kararlılığını ($\Delta\Delta G$) canlı olarak hesaplayarak klinikteki "Yanlış Negatif (False Negative)" vakalarını asgariye indirir.

---

## ✨ Öne Çıkan Özellikler

- 🔍 **Kademeli Hibrit Güvenlik Ağı:** Standart missense modelinin "Zararsız" dediği varyantlar eve gönderilmez. İkinci bir güvenlik aşamasında "Salt Sessiz Modülü" üzerinden RNA ve evrimsel analizlere sokularak gizli patojenite aranır.
- 🧬 **Canlı Özellik Çıkarımı:** ClinVar ve gnomAD veritabanları entegrasyonu sayesinde, verilen herhangi bir varyantın 12 biyolojik özelliği (RSCU, CAI, phyloP, phastCons vb.) saniyeler içinde API'ler aracılığıyla canlı çekilir ve hesaplanır.
- ⚖️ **Sınıf Dengesizliği (Class Imbalance) Yönetimi:** Genomik verilerdeki devasa hasta/sağlıklı dengesizliği; KNN-Imputer, RobustScaler, Youden İndeksi tabanlı optimal klinik karar eşiği (Threshold Optimization) ve Gen-bazlı (GroupK-Fold) çapraz doğrulama ile çözülmüştür.
- 💡 **Açıklanabilir Yapay Zeka (XAI):** Oyun teorisi tabanlı **SHAP** entegrasyonu sayesinde model kara kutu olmaktan çıkarılmıştır. Her bir hastanın varyantında, hangi biyolojik özelliğin kararı yüzde kaç etkilediği doktora şeffaf bir grafikle sunulur.

---

## 🚀 Kurulum

Projeyi yerel makinenizde (lokalde) çalıştırmak için aşağıdaki adımları izleyin:

**1. Depoyu Klonlayın:**
```bash
git clone https://github.com/KULLANICI_ADINIZ/GenFlow.git
cd GenFlow
```

**2. Sanal Ortam (Virtual Environment) Oluşturun ve Aktif Edin:**
```bash
python -m venv venv
# Windows için:
venv\Scripts\activate
# Linux/MacOS için:
source venv/bin/activate
```

**3. Gerekli Kütüphaneleri Yükleyin:**
```bash
pip install -r requirements.txt
```

---

## 💻 Kullanım (Dashboard)

Geliştirilen sistem komut satırında çalışan sıradan bir script değil; klinisyenlerin doğrudan kullanabileceği, interaktif ve modern bir web arayüzüne sahiptir.

Arayüzü başlatmak için:
```bash
streamlit run src/dashboard.py
```
*(Uygulama çalıştıktan sonra tarayıcınızda `http://localhost:8501` adresinden erişebilirsiniz.)*

**Dashboard Sekmeleri:**
1. **Model Açıklanabilirliği (SHAP):** Yapay zekanın patojenite kararını nasıl aldığını şeffafça inceleyin.
2. **Klinik Optimizasyon & Karar Eşiği:** Youden İndeksi tabanlı stres testini ve duyarlılık (Recall) eğrilerini görüntüleyin.
3. **Özel Senaryo (What-If):** Patojenik bir hastayı alıp genetik özellikleriyle oynayarak riski nasıl düşürebileceğinizi test edin.
4. **Salt Sessiz (Synonymous) Mutasyon Analizi:** Herhangi bir varyant kodunu girerek mRNA stabilitesi ve Kodon analizini canlı gerçekleştirin.
5. **Kademeli Hibrit Sistem:** Missense modelini geçen varyantların RNA analizine otomatik girmesini simüle eden 2 aşamalı güvenlik ağı.

---

## 📂 Proje Yapısı

```text
GenFlow/
├── data/                    # İşlenmiş ve ham genomik veriler (CSV)
├── models/                  # Eğitilmiş XGBoost modelleri (.pkl) ve eşik değerleri
├── src/
│   ├── custom_model.py      # XGBoost ve model mimarisi sınıfları
│   ├── feature_extractor.py # Dış API'lerden canlı veri ve biyolojik skor üreten modül
│   ├── dashboard.py         # Streamlit web arayüzü (Klinisyen paneli)
│   └── model_trainer.py     # Optuna tabanlı hiperparametre optimizasyonu ve GroupKFold eğitim kodu
├── tests/                   # Birim testleri
├── requirements.txt         # Proje bağımlılıkları
└── README.md                # Proje dokümantasyonu
```

---

## 📊 Sonuçlar ve Performans Metrikleri

XGBoost ile Bayesian Optimizasyon kullanılarak geliştirilen model, asimetrik sınıf dağılımına rağmen yüksek başarı elde etmiştir. **F1 Skoru** ve **Matthews Korelasyon Katsayısı (MCC)** metrikleri özellikle referans alınmıştır.

| Varyant Modülü | Alt Grup | PR-AUC | F1-Skoru | MCC |
| :--- | :--- | :--- | :--- | :--- |
| **Missense** | Genel Kapsam (MASTER) | 0.934 | 0.912 | 0.895 |
| **Synonymous**| Supervised Model | **0.967** | **0.895** | **0.780** |

*Özellikle Synonymous (Sessiz) varyant modelimiz, Youden optimal eşiği (0.495) sayesinde %91.83 Recall (Duyarlılık) oranına ulaşarak klinik literatürdeki yanlış negatifleri devasa oranda engellemiştir.*

---

## 🏆 TEKNOFEST Bilgilendirmesi
Bu proje **TEKNOFEST Sağlıkta Yapay Zeka Yarışması** finali için geliştirilmiştir. Raporlar, sunumlar ve kodlama standartları tamamen açık kaynak prensiplerine ve tıbbi araştırma etiğine uygun şekilde dizayn edilmiştir.