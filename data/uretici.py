"""
LOTHUS TAKIMI — İZOLE SESSİZ MUTASYON VERİ FABRİKASI
"""
import pandas as pd
import numpy as np
import os

def gen_dosyalarini_isle(txt_listesi, cikti_adi, sablon_kolonlar):
    df_listesi = []
    for txt in txt_listesi:
        if not os.path.exists(txt):
            print(f"⚠️ Atlandı: {txt} bulunamadı.")
            continue
        # gnomAD txt dosyalarını TAB ayrımıyla okur
        df = pd.read_csv(txt, sep="\t") 
        df_listesi.append(df)
        
    if not df_listesi: return

    # Dosyaları alt alta birleştir (Örn: BRCA1 ve BRCA2)
    birlesik_df = pd.concat(df_listesi, ignore_index=True)

    # Sözlük (Mapping) İşlemi
    birlesik_df = birlesik_df.rename(columns={
        "gnomAD ID": "Variant_ID",
        "Allele Frequency": "AL_1"
    })

    # 354 kolonluk şablonu uygula
    final_df = pd.DataFrame(columns=sablon_kolonlar)
    for col in sablon_kolonlar:
        if col in birlesik_df.columns:
            final_df[col] = birlesik_df[col]
        else:
            final_df[col] = np.nan 

    final_df["Label"] = 0 

    final_df.to_csv(cikti_adi, index=False)
    print(f"✅ ÜRETİLDİ: {cikti_adi} (Toplam {len(final_df)} varyant)")

def main():
    print("🚀 İzole Üretim Fabrikası çalışıyor...\n")

    try:
        sablon_df = pd.read_csv("processed_kanser.csv")
        sablon_kolonlar = sablon_df.columns.tolist()
    except FileNotFoundError:
        print("❌ HATA: 'processed_kanser.csv' şablonu bu klasörde bulunamadı!")
        return

    # Görev 1: Kanser (BRCA1 ve BRCA2 birleşiyor)
    gen_dosyalarini_isle(["BRCA1_gnomad_clean.txt", "BRCA2_gnomad_clean.txt"], "processed_sessiz_mutasyon_kanser.csv", sablon_kolonlar)
    
    # Görev 2: MAPT
    gen_dosyalarini_isle(["MAPT_gnomad_clean.txt"], "processed_sessiz_mutasyon_mapt.csv", sablon_kolonlar)
    
    # Görev 3: CFTR
    gen_dosyalarini_isle(["CFTR_gnomad_clean.txt"], "processed_sessiz_mutasyon_cftr.csv", sablon_kolonlar)
    
    # Görev 4: PAH
    gen_dosyalarini_isle(["PAH_gnomad_clean.txt"], "processed_sessiz_mutasyon_pah.csv", sablon_kolonlar)

    # Görev 5: MASTER ve GENEL
    tum_txtler = ["BRCA1_gnomad_clean.txt", "BRCA2_gnomad_clean.txt", "MAPT_gnomad_clean.txt", "CFTR_gnomad_clean.txt", "PAH_gnomad_clean.txt"]
    gen_dosyalarini_isle(tum_txtler, "processed_sessiz_mutasyon_master.csv", sablon_kolonlar)
    pd.read_csv("processed_sessiz_mutasyon_master.csv").to_csv("processed_sessiz_mutasyon_genel.csv", index=False)

    print("\n🎉 Bütün dosyalar başarıyla hazırlandı. Şimdi .csv dosyalarını asıl projeye taşıma vakti!")

if __name__ == "__main__":
    main()