"""
GEN FLOW — TEKNOFEST Sağlıkta Yapay Zeka Yarışması
Uçtan uca veri temizleme ve özellik mühendisliği boru hattı (pipeline).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder

EPSILON: float = 1e-8

# ARTIK AA_1 VE AA_2 BURADA DEĞİL! Sadece Variant_ID model dışı kalacak.
METADATA_COLUMNS: tuple[str, ...] = ("Variant_ID",)
LABEL_COLUMN: str = "Label"

class GenFlowPreprocessor:
    def __init__(self, csv_path: str | Path, max_iter: int = 10, random_state: int = 42) -> None:
        self.csv_path = Path(csv_path)
        self.max_iter = max_iter
        self.random_state = random_state

        self._imputer: Optional[IterativeImputer] = None
        self._scaler: Optional[RobustScaler] = None
        self._encoder: Optional[OrdinalEncoder] = None

        self.metadata_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.numerical_columns: list[str] = []
        self.al_columns: list[str] = []
        self.ek_columns: list[str] = []

        self._raw_df: Optional[pd.DataFrame] = None
        self._processed_df: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV dosyası bulunamadı: {self.csv_path}")
        self._raw_df = pd.read_csv(self.csv_path)
        return self._raw_df

    def split_columns(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        all_columns = df.columns.tolist()

        self.metadata_columns = [c for c in METADATA_COLUMNS if c in all_columns]

        # YENİ ZEKAMIZ BURADA: AA_1, AA_2 ve tüm CAT_ kolonları kategorik olarak ayrılıyor
        self.categorical_columns = [
            c for c in all_columns 
            if c.startswith("CAT_") or c in ("AA_1", "AA_2")
        ]

        self.al_columns = [c for c in all_columns if c.startswith("AL_")]
        self.ek_columns = [c for c in all_columns if c.startswith("EK_")]

        excluded = set(self.metadata_columns + self.categorical_columns + ([LABEL_COLUMN] if LABEL_COLUMN in all_columns else []))
        self.numerical_columns = [c for c in all_columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]

        groups: dict[str, pd.DataFrame] = {
            "metadata": df[self.metadata_columns].copy() if self.metadata_columns else pd.DataFrame(index=df.index),
            "categorical": df[self.categorical_columns].astype(str).copy() if self.categorical_columns else pd.DataFrame(index=df.index),
            "numerical": df[self.numerical_columns].copy(),
        }

        if LABEL_COLUMN in all_columns:
            groups["label"] = df[[LABEL_COLUMN]].copy()

        return groups

    def impute_numerical(self, df_numerical: pd.DataFrame) -> pd.DataFrame:
        if df_numerical.empty:
            return df_numerical
        self._imputer = IterativeImputer(max_iter=self.max_iter, random_state=self.random_state)
        imputed_array = self._imputer.fit_transform(df_numerical)
        return pd.DataFrame(imputed_array, columns=df_numerical.columns, index=df_numerical.index)

    def encode_categorical(self, df_categorical: pd.DataFrame) -> pd.DataFrame:
        """Harfleri (Kategorik verileri) sayılara dönüştürür."""
        if df_categorical.empty:
            return df_categorical
        # Bilinmeyen bir harf/gen gelirse patlamasın diye -1 atayacak
        self._encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        encoded_array = self._encoder.fit_transform(df_categorical)
        return pd.DataFrame(encoded_array, columns=df_categorical.columns, index=df_categorical.index)

    @staticmethod
    def create_al_ek_interaction(df_numerical: pd.DataFrame, al_columns: list[str], ek_columns: list[str], epsilon: float = EPSILON) -> pd.Series:
        if not al_columns or not ek_columns:
            raise ValueError("AL_EK_Interaction için hem AL_ hem de EK_ kolonları gerekli.")
        al_mean = df_numerical[al_columns].mean(axis=1)
        ek_mean = df_numerical[ek_columns].mean(axis=1)
        interaction = al_mean / (ek_mean + epsilon)
        return interaction.rename("AL_EK_Interaction")

    def scale_numerical(self, df_numerical: pd.DataFrame) -> pd.DataFrame:
        if df_numerical.empty:
            return df_numerical
        self._scaler = RobustScaler()
        scaled_array = self._scaler.fit_transform(df_numerical)
        return pd.DataFrame(scaled_array, columns=df_numerical.columns, index=df_numerical.index)

    def transform(self) -> pd.DataFrame:
        raw_df = self.load_data()
        groups = self.split_columns(raw_df)

        # 1. Sayısal kolonları doldur
        df_imputed = self.impute_numerical(groups["numerical"])
        
        # 2. Sentetik özellik üret
        interaction = self.create_al_ek_interaction(df_imputed, al_columns=self.al_columns, ek_columns=self.ek_columns)
        df_imputed["AL_EK_Interaction"] = interaction
        
        # 3. Sayısalları ölçeklendir
        df_scaled = self.scale_numerical(df_imputed)

        # 4. Kategorik (Harfli) kolonları sayısallaştır
        df_encoded = self.encode_categorical(groups["categorical"])

        parts: list[pd.DataFrame] = []
        if not groups["metadata"].empty:
            parts.append(groups["metadata"])
        parts.append(df_scaled)
        if not df_encoded.empty:
            parts.append(df_encoded)
        if "label" in groups:
            parts.append(groups["label"])

        self._processed_df = pd.concat(parts, axis=1)
        return self._processed_df

    def save_processed(self, output_path: str | Path) -> None:
        if self._processed_df is None:
            raise RuntimeError("Kaydedilecek veri yok. Önce transform() çalıştırın.")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._processed_df.to_csv(output_path, index=False)

# ======================================================================
# Toplu Çalıştırma Senaryosu — Tüm Veri Setleri
# ======================================================================
if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / "data"

    datasets = [
        "YARISMA_TRAIN_MASTER.csv",
        "YARISMA_TRAIN_KANSER.csv",
        "YARISMA_TRAIN_PAH.csv",
        "YARISMA_TRAIN_CFTR.csv"
    ]

    print("=" * 60)
    print("GEN FLOW — Toplu Veri Ön İşleme (Kategorik Zeka Eklendi)")
    print("=" * 60)

    for dataset_name in datasets:
        input_csv = DATA_DIR / dataset_name
        clean_name = dataset_name.replace("YARISMA_TRAIN_", "")
        output_csv = DATA_DIR / f"processed_{clean_name}"

        if not input_csv.exists():
            print(f"⚠️ Uyarı: {dataset_name} bulunamadı, atlanıyor...")
            continue

        print(f"\n[İŞLENİYOR] -> {dataset_name}")
        preprocessor = GenFlowPreprocessor(csv_path=input_csv)
        processed_df = preprocessor.transform()
        preprocessor.save_processed(output_csv)

        print(f"✅ Başarılı! Çıktı: {output_csv.name} "
              f"(Satır: {len(processed_df)}, Kolon: {processed_df.shape[1]})")

    print("\n" + "=" * 60)
    print("Tüm veri setleri kategorik verilerle birlikte işlendi!")
    print("=" * 60)