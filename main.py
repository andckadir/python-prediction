import re
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Yapay Zeka Analiz Paneli API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Modelleri ve Scaler'ları Yükleme
# ---------------------------------------------------------------------------
# Diabetes (Classification)
diabetes_model = joblib.load("models/diabetes_model.pkl")
diabetes_scaler = joblib.load("models/diabetes_scaler.pkl")
# Median degerleri: preprocessing raporunda 0 -> NaN -> median ile dolduruluyordu.
# Egitim sirasinda hesaplanan medianlari da kaydedip yuklemeniz gerekiyor, ornek:
#   medians = {"Glucose": 117.0, "BloodPressure": 72.0, "SkinThickness": 23.0, "BMI": 32.0}
#   joblib.dump(medians, "models/diabetes_medians.pkl")
diabetes_medians = joblib.load("models/diabetes_medians.pkl")

DIABETES_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
# Preprocessing raporunda 0 degeri anlamsiz kabul edilen kolonlar (Insulin haric,
# o zaten ARM/Classification akisinda ayri ele aliniyor)
ZERO_AS_MISSING_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "BMI"]

# Digits (Classification - multiclass)
digits_model = joblib.load("models/digits_model.pkl")
digits_scaler = joblib.load("models/digits_scaler.pkl")

# RFM / K-Means (Clustering)
kmeans_model = joblib.load("models/kmeans_model.pkl")
rfm_scaler = joblib.load("models/rfm_scaler.pkl")

# Clustering raporundaki centroid yorumuna gore doldurun (0..3 -> anlamli etiket)
CLUSTER_LABELS = {
    0: "Sadık / Yüksek Değerli Müşteri",
    1: "Riskli / Kaybedilme Tehlikesi Olan Müşteri",
    2: "Yeni / Az Etkileşimli Müşteri",
    3: "Ortalama Müşteri",
}

# Association Rule Mining - onceden hesaplanmis kural tablosu
ecommerce_rules = pd.read_csv("models/ecommerce_rules_france.csv")


def _parse_frozenset_column(series: pd.Series) -> pd.Series:
    """'frozenset({\\'A\\', \\'B\\'})' seklinde CSV'den okunan string kolonlari
    Python listesine cevirir."""
    def parse(value: str):
        return re.findall(r"'([^']*)'", str(value))
    return series.apply(parse)


ecommerce_rules["antecedents_list"] = _parse_frozenset_column(ecommerce_rules["antecedents"])
ecommerce_rules["consequents_list"] = _parse_frozenset_column(ecommerce_rules["consequents"])


# ---------------------------------------------------------------------------
# Pydantic Şemaları
# ---------------------------------------------------------------------------
class DiabetesInput(BaseModel):
    data: list[float]  # 8 deger: DIABETES_COLUMNS sirasiyla


class DigitInput(BaseModel):
    pixels: list[float]  # 64 deger (8x8, 0-16 araligi)


class RFMInput(BaseModel):
    recency: float
    frequency: float
    monetary: float


class ProductInput(BaseModel):
    product: str


# ---------------------------------------------------------------------------
# 1) CLASSIFICATION - Diyabet
# ---------------------------------------------------------------------------
@app.post("/predict/diabetes")
async def predict_diabetes(input: DiabetesInput):
    if len(input.data) != 8:
        raise HTTPException(status_code=400, detail="8 özellik bekleniyor (Insulin dahil).")
    scaled = diabetes_scaler.transform([input.data])
    pred = diabetes_model.predict(scaled)
    prob = diabetes_model.predict_proba(scaled)[0][1]
    return {"prediction": int(pred[0]), "probability": float(prob) * 100}


# ---------------------------------------------------------------------------
# 2) CLASSIFICATION - El Yazısı Rakam
# ---------------------------------------------------------------------------
@app.post("/predict/digit")
async def predict_digit(input: DigitInput):
    if len(input.pixels) != 64:
        raise HTTPException(status_code=400, detail="64 piksel bekleniyor.")
    scaled = digits_scaler.transform([input.pixels])
    pred = digits_model.predict(scaled)
    proba = digits_model.predict_proba(scaled)[0]
    return {"prediction": int(pred[0]), "confidence": float(max(proba)) * 100}


# ---------------------------------------------------------------------------
# 3) CLUSTERING - RFM / K-Means
# ---------------------------------------------------------------------------
@app.post("/cluster/rfm")
async def cluster_rfm(input: RFMInput):
    rfm_log = np.log1p([input.recency, input.frequency, input.monetary])
    scaled = rfm_scaler.transform([rfm_log])
    cluster = int(kmeans_model.predict(scaled)[0])
    return {"cluster": cluster, "label": CLUSTER_LABELS.get(cluster, f"Küme {cluster}")}


# ---------------------------------------------------------------------------
# 4) DATA PREPROCESSING - Diyabet verisi üzerinde canlı önizleme
# ---------------------------------------------------------------------------
@app.post("/preprocess/diabetes")
async def preprocess_diabetes(input: DiabetesInput):
    if len(input.data) != 8:
        raise HTTPException(status_code=400, detail="8 özellik bekleniyor (Insulin dahil).")

    raw = dict(zip(DIABETES_COLUMNS, input.data))
    imputed = dict(raw)
    imputed_flags = {}

    # 0 -> median imputation (sadece raporda belirtilen kolonlarda)
    for col in ZERO_AS_MISSING_COLUMNS:
        if imputed[col] == 0:
            imputed[col] = diabetes_medians[col]
            imputed_flags[col] = True
        else:
            imputed_flags[col] = False

    ordered_values = [imputed[c] for c in DIABETES_COLUMNS]
    scaled_values = diabetes_scaler.transform([ordered_values])[0]
    scaled = dict(zip(DIABETES_COLUMNS, [round(float(v), 4) for v in scaled_values]))

    return {
        "raw": raw,
        "imputed": imputed,
        "imputed_flags": imputed_flags,
        "scaled": scaled,
    }


# ---------------------------------------------------------------------------
# 5) ASSOCIATION RULE MINING - Market Basket (Online Retail / France)
# ---------------------------------------------------------------------------
@app.get("/rules/products")
async def list_products(limit: int = 60):
    """Kural tablosunda gecen urunlerin listesini dondurur (dropdown/autocomplete icin)."""
    all_products = set()
    for lst in ecommerce_rules["antecedents_list"]:
        all_products.update(lst)
    for lst in ecommerce_rules["consequents_list"]:
        all_products.update(lst)
    return {"products": sorted(all_products)[:limit]}


@app.post("/rules/recommend")
async def recommend_products(input: ProductInput):
    query = input.product.strip().upper()
    if not query:
        raise HTTPException(status_code=400, detail="Ürün adı boş olamaz.")

    matches = ecommerce_rules[
        ecommerce_rules["antecedents_list"].apply(lambda lst: query in lst)
    ].sort_values(by="confidence", ascending=False)

    if matches.empty:
        return {"query": query, "recommendations": []}

    recommendations = []
    for _, row in matches.head(5).iterrows():
        recommendations.append({
            "consequent": ", ".join(row["consequents_list"]),
            "support": round(float(row["support"]), 4),
            "confidence": round(float(row["confidence"]), 4),
            "lift": round(float(row["lift"]), 4),
        })

    return {"query": query, "recommendations": recommendations}
