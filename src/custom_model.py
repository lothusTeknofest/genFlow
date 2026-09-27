"""
GEN FLOW — TEKNOFEST Sağlıkta Yapay Zeka Yarışması
Özel makine öğrenmesi model çekirdeği.

Temel yapı:
    XGBClassifier + Asimetrik Focal Loss + Isotonic Kalibrasyon
Hedef: Patojenik (1) / Benign (0) sınıflandırması
"""

from __future__ import annotations

from functools import partial
from typing import Any, Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Tıbbi maliyet sabitleri
# ---------------------------------------------------------------------------
# False Negative (Patojenik → Benign): hasta risk altında kalır → 4x ağır
FN_PENALTY: float = 4.0
# False Positive (Benign → Patojenik): gereksiz takip → referans maliyet
FP_PENALTY: float = 1.0
# Focal Loss gamma: zor örneklerin gradyan ağırlığını artırır (standart = 2)
FOCAL_GAMMA: float = 2.0
# Olasılık hesaplarında log(0) ve bölme hatalarını önleyen alt/üst sınır
PROB_CLIP_EPS: float = 1e-7


def asymmetric_focal_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gamma: float = FOCAL_GAMMA,
    fn_weight: float = FN_PENALTY,
    fp_weight: float = FP_PENALTY,
) -> tuple[np.ndarray, np.ndarray]:
    """
    XGBoost 'objective' parametresine gömülen Özel Asimetrik Focal Loss.

    Tıbbi maliyet mantığı:
        - Gerçek Patojenik (y=1) iken düşük olasılık vermek (FN) → fn_weight=4 ile cezalandırılır.
        - Gerçek Benign (y=0) iken yüksek olasılık vermek (FP) → fp_weight=1 ile cezalandırılır.

    Focal modülasyonu (1-p)^γ / p^γ: kolay örneklerin gradyanını bastırır,
    sınırda kalan (zor) varyantlara odaklanmayı sağlar.

    Parametreler
    ----------
    y_true : np.ndarray
        Gerçek sınıf etiketleri (0: Benign, 1: Patojenik).
        XGBoost 3.x sklearn API imzası: objective(y_true, y_pred).
    y_pred : np.ndarray
        XGBoost'un ürettiği ham marj (logit) tahminleri.
    gamma : float
        Focal Loss odaklanma parametresi.
    fn_weight : float
        False Negative (y=1, p düşük) ceza katsayısı.
    fp_weight : float
        False Positive (y=0, p yüksek) ceza katsayısı.

    Dönüş
    -----
    grad, hess : np.ndarray
        XGBoost ağaç bölünmesi için birinci ve ikinci türev vektörleri.
    """
    # XGBoost 3.x: y_true doğrudan ilk argüman olarak gelir
    y_true = np.asarray(y_true)

    # Ham marj → olasılık dönüşümü (sigmoid)
    prob = 1.0 / (1.0 + np.exp(-y_pred))
    # Olasılıkları güvenli aralığa kırp (log ve bölme için)
    prob = np.clip(prob, PROB_CLIP_EPS, 1.0 - PROB_CLIP_EPS)

    # Sık kullanılan ara terimler
    one_minus_prob = 1.0 - prob          # (1 - p)
    log_prob = np.log(prob)              # log(p)
    log_one_minus_prob = np.log(one_minus_prob)  # log(1 - p)

    # --- y=1 (Patojenik) dalı: L₁ = -fn_weight · (1-p)^γ · log(p) ---
    # dL₁/dp = fn_weight · [(1-p)^γ/p + γ·(1-p)^(γ-1)·log(p)]
    dloss_pos = fn_weight * (
        (one_minus_prob ** gamma) / prob
        + gamma * (one_minus_prob ** (gamma - 1)) * log_prob
    )

    # --- y=0 (Benign) dalı: L₀ = -fp_weight · p^γ · log(1-p) ---
    # dL₀/dp = fp_weight · [p^γ/(1-p) - γ·p^(γ-1)·log(1-p)]
    dloss_neg = fp_weight * (
        (prob ** gamma) / one_minus_prob
        - gamma * (prob ** (gamma - 1)) * log_one_minus_prob
    )

    # Etikete göre doğru dalın gradyanını seç
    dloss_dprob = np.where(y_true == 1, dloss_pos, dloss_neg)

    # Zincir kuralı: dL/d(logit) = dL/dp · dp/d(logit),  dp/d(logit) = p(1-p)
    grad = dloss_dprob * prob * one_minus_prob

    # İkinci türev yaklaşımı — XGBoost yaprağı bölme kararları için yeterli
    hess = np.maximum(grad * (1.0 - grad), 1e-6)

    return grad, hess


class GenFlowModel:
    """
    GEN FLOW sınıflandırma modeli.

    Mimari:
        1. XGBClassifier  → temel gradyan artırma ağacı
        2. Asimetrik Focal Loss → FN maliyetini modele enjekte eder
        3. CalibratedClassifierCV (isotonic, cv=3) → olasılık kalibrasyonu;
           jüri test setindeki dağılım kaymasına (data shift) karşı koruma sağlar.

    Kullanım (train.py'de tamamlanacak):
        model = GenFlowModel()
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
    """

    def __init__(
        self,
        fn_penalty: float = FN_PENALTY,
        fp_penalty: float = FP_PENALTY,
        gamma: float = FOCAL_GAMMA,
        calibration_cv: int = 3,
        random_state: int = 42,
        **xgb_params: Any,
    ) -> None:
        """
        Parametreler
        ----------
        fn_penalty : float
            False Negative ceza katsayısı (varsayılan 4× FP maliyeti).
        fp_penalty : float
            False Positive ceza katsayısı (referans = 1).
        gamma : float
            Focal Loss odaklanma parametresi.
        calibration_cv : int
            Isotonic kalibrasyon için çapraz doğrulama kat sayısı.
        random_state : int
            Tekrarlanabilirlik için rastgele tohum.
        **xgb_params : Any
            XGBClassifier'a iletilecek ek hiperparametreler (n_estimators, max_depth vb.).
        """
        # Tıbbi maliyet katsayılarını sakla
        self.fn_penalty = fn_penalty
        self.fp_penalty = fp_penalty
        self.gamma = gamma
        self.calibration_cv = calibration_cv
        self.random_state = random_state
        self.xgb_params = xgb_params

        # Asimetrik Focal Loss'u sabit hiperparametrelerle partial olarak bağla
        # (XGBoost 3.x objective imzası: fn(y_true, y_pred) → (grad, hess))
        self._focal_objective = partial(
            asymmetric_focal_loss,
            gamma=self.gamma,
            fn_weight=self.fn_penalty,
            fp_weight=self.fp_penalty,
        )

        # XGBoost temel sınıflandırıcı — özel kayıp fonksiyonu objective'e gömülü
        self._base_classifier = XGBClassifier(
            objective=self._focal_objective,   # Özel Asimetrik Focal Loss
            eval_metric="logloss",             # Eğitim izleme metriği
            random_state=self.random_state,    # Tekrarlanabilirlik
            n_jobs=-1,                         # Tüm CPU çekirdeklerini kullan
            **self.xgb_params,                 # Kullanıcı tanımlı ek parametreler
        )

        # Isotonic kalibrasyon sarmalayıcısı — dağılım kaymasına karşı olasılık düzeltmesi
        self._calibrated_model = CalibratedClassifierCV(
            estimator=self._base_classifier,   # Kalibre edilecek temel model
            method="isotonic",                 # Monoton olasılık eşlemesi
            cv=self.calibration_cv,            # 3 katlı iç çapraz doğrulama
        )

        # fit() çağrıldıktan sonra True olur
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Eğitim
    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray | Any,
        y: np.ndarray | Any,
    ) -> "GenFlowModel":
        """
        Modeli eğitir: XGBoost + Isotonic kalibrasyon.

        Parametreler
        ----------
        X : array-like, shape (n_samples, n_features)
            Ön işlenmiş özellik matrisi.
        y : array-like, shape (n_samples,)
            Hedef etiketler (0: Benign, 1: Patojenik).

        Dönüş
        -----
        self : GenFlowModel
            Zincirleme çağrı (method chaining) için kendisi.
        """
        # Kalibre edilmiş modeli eğit (iç CV ile isotonic eşleme öğrenilir)
        self._calibrated_model.fit(X, y)
        # Eğitim tamamlandı bayrağını güncelle
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Olasılık tahmini
    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray | Any) -> np.ndarray:
        """
        Kalibre edilmiş sınıf olasılıklarını döndürür.

        Parametreler
        ----------
        X : array-like, shape (n_samples, n_features)
            Tahmin yapılacak özellik matrisi.

        Dönüş
        -----
        proba : np.ndarray, shape (n_samples, 2)
            [:, 0] → Benign (0) olasılığı
            [:, 1] → Patojenik (1) olasılığı  ← tıbbi karar için birincil çıktı

        Raises
        ------
        RuntimeError
            fit() henüz çağrılmadıysa.
        """
        # Eğitimsiz modelde tahmin yapılmasını engelle
        if not self._is_fitted:
            raise RuntimeError(
                "Model henüz eğitilmedi. Önce fit(X, y) metodunu çağırın."
            )

        # Isotonic kalibrasyon uygulanmış olasılık vektörünü döndür
        return self._calibrated_model.predict_proba(X)

    # ------------------------------------------------------------------
    # Yardımcı özellikler
    # ------------------------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        """Modelin eğitilip eğitilmediğini bildirir."""
        return self._is_fitted

    @property
    def calibrated_estimator(self) -> CalibratedClassifierCV:
        """Alt sklearn sarmalayıcısına doğrudan erişim (train.py / analiz için)."""
        return self._calibrated_model
    
import numpy as np
from sklearn.ensemble import IsolationForest

class SessizMutasyonModel:
    """Sessiz (Synonymous) varyantlar için Anomali Tespit Sınıfı."""
    def __init__(self, contamination=0.01, random_state=42):
        self.model = IsolationForest(contamination=contamination, random_state=random_state)
        self._is_fitted = False
        self.optimal_threshold_ = 0.5 

    def fit(self, X):
        print("🧠 Sessiz Mutasyonlar (Isolation Forest) eğitiliyor...")
        self.model.fit(X)
        self._is_fitted = True
        return self

    def predict(self, X):
        if not self._is_fitted: raise ValueError("Model henüz eğitilmedi!")
        preds = self.model.predict(X)
        return np.where(preds == -1, 1, 0)

    def predict_proba(self, X):
        """
        Arayüzün 0-100 arasında klinik (sürekli) bir risk skoru üretebilmesi için,
        varyantın sağlıklı popülasyona olan uzaklık skorunu Sigmoid eğrisi ile olasılığa büker.
        """
        if not self._is_fitted: 
            return np.array([[0.5, 0.5]] * len(X))
        
        # decision_function: Pozitif = Sağlıklı Merkeze Yakın, Negatif = Anomali (Uzak)
        scores = self.model.decision_function(X)
        
        # Sigmoid dönüşümü: Anomali uzaklığını 0.0 ile 1.0 arasında bir yüzdelik prime çevirir.
        # k=15 faktörü, geçişi klinik olarak yumuşatır.
        k = 15 
        risk = 1 / (1 + np.exp(scores * k))
        
        proba = np.zeros((len(X), 2))
        proba[:, 1] = risk       # Patojenik (Anomali) Olasılığı
        proba[:, 0] = 1 - risk   # Benign (Sağlıklı) Olasılığı
        return proba