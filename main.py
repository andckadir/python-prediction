import os
import re
import io
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

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

# Median degerleri
if os.path.exists("models/diabetes_medians.pkl"):
    diabetes_medians = joblib.load("models/diabetes_medians.pkl")
else:
    diabetes_medians = {"Glucose": 117.0, "BloodPressure": 72.0, "SkinThickness": 29.0, "BMI": 32.3}

DIABETES_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
ZERO_AS_MISSING_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "BMI"]

# Digits (Classification - multiclass)
digits_model = joblib.load("models/digits_model.pkl")
digits_scaler = joblib.load("models/digits_scaler.pkl")
digits_dataset = load_digits()

# RFM / K-Means (Clustering)
kmeans_model = joblib.load("models/kmeans_model.pkl")
rfm_scaler = joblib.load("models/rfm_scaler.pkl")
rfm_data = joblib.load("models/rfm_data.pkl") if os.path.exists("models/rfm_data.pkl") else None
if os.path.exists("models/rfm_elbow_wcss.pkl"):
    rfm_elbow_wcss = joblib.load("models/rfm_elbow_wcss.pkl")
else:
    rfm_elbow_wcss = [12936.0, 6598.96, 4957.95, 3943.33, 3296.19, 2893.23, 2574.44, 2369.21, 2182.59, 2025.29]

# Clustering etiketleri (Centroid analizine gore)
CLUSTER_LABELS = {
    0: "Ortalama / Düzenli Müşteri",
    1: "Riskli / Kaybedilme Tehlikesi Olan Müşteri",
    2: "Sadık / Yüksek Değerli Müşteri (VIP)",
    3: "Yeni / Az Etkileşimli Müşteri",
}

# Association Rule Mining - onceden hesaplanmis kural tablosu
rules_path = "models/ecommerce_rules_france.csv" if os.path.exists("models/ecommerce_rules_france.csv") else "ecommerce_rules_france.csv"
ecommerce_rules = pd.read_csv(rules_path)


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
    input_df = pd.DataFrame([input.data], columns=DIABETES_COLUMNS)
    scaled = diabetes_scaler.transform(input_df)
    pred = diabetes_model.predict(scaled)
    prob = diabetes_model.predict_proba(scaled)[0][1]
    return {"prediction": int(pred[0]), "probability": float(prob) * 100}


# ---------------------------------------------------------------------------
# 2) CLASSIFICATION - El Yazısı Rakam
# ---------------------------------------------------------------------------
@app.get("/digits/sample")
async def get_digit_sample(digit: int | None = None):
    if digit is not None and (digit < 0 or digit > 9):
        raise HTTPException(status_code=400, detail="Rakam 0 ile 9 arasında olmalıdır.")

    if digit is not None:
        indices = np.where(digits_dataset.target == digit)[0]
    else:
        indices = np.arange(len(digits_dataset.target))

    if len(indices) == 0:
        raise HTTPException(status_code=404, detail="Örnek bulunamadı.")

    chosen_idx = int(np.random.choice(indices))
    pixels = [float(v) for v in digits_dataset.data[chosen_idx]]
    target = int(digits_dataset.target[chosen_idx])
    return {
        "digit": target,
        "pixels": pixels,
        "index": chosen_idx
    }


@app.post("/predict/digit")
async def predict_digit(input: DigitInput):
    if len(input.pixels) != 64:
        raise HTTPException(status_code=400, detail="64 piksel bekleniyor.")
    scaled = digits_scaler.transform([input.pixels])
    pred = int(digits_model.predict(scaled)[0])
    proba = digits_model.predict_proba(scaled)[0]
    probabilities = {int(cls): round(float(prob) * 100, 2) for cls, prob in zip(digits_model.classes_, proba)}
    return {
        "prediction": pred,
        "confidence": round(float(max(proba)) * 100, 2),
        "probabilities": probabilities
    }


# ---------------------------------------------------------------------------
# 3) CLUSTERING - RFM / K-Means
# ---------------------------------------------------------------------------
@app.post("/cluster/rfm")
async def cluster_rfm(input: RFMInput):
    rfm_log = np.log1p([input.recency, input.frequency, input.monetary])
    input_df = pd.DataFrame([rfm_log], columns=["Recency", "Frequency", "Monetary"])
    scaled = rfm_scaler.transform(input_df)
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
    input_df = pd.DataFrame([ordered_values], columns=DIABETES_COLUMNS)
    scaled_values = diabetes_scaler.transform(input_df)[0]
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


# ---------------------------------------------------------------------------
# 6) MODEL EVALUATION & VISUALIZATION (Rapordaki Confusion Matrix Dağılımları)
# ---------------------------------------------------------------------------
def generate_diabetes_cm_image():
    csv_path = 'csvs/diabetes.csv' if os.path.exists('csvs/diabetes.csv') else 'diabetes.csv'
    df = pd.read_csv(csv_path, header=None)
    df.columns = DIABETES_COLUMNS + ['Outcome']
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test_scaled = diabetes_scaler.transform(X_test)
    y_pred = diabetes_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(7.2, 5.4), facecolor='#0f172a')
    ax = plt.gca()
    ax.set_facecolor('#0f172a')
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Greens', cbar=False,
        xticklabels=['Healthy (0)', 'Diabetic (1)'],
        yticklabels=['Healthy (0)', 'Diabetic (1)'],
        annot_kws={"size": 18, "weight": "bold"}
    )
    plt.title('Confusion Matrix Heatmap (Diabetes Dataset)', fontsize=14, fontweight='bold', color='#2dd4bf', pad=14)
    plt.ylabel('True Label (Gerçek Değer)', fontweight='bold', color='#cbd5e1', fontsize=12, labelpad=8)
    plt.xlabel('Predicted Label (Model Tahmini)', fontweight='bold', color='#cbd5e1', fontsize=12, labelpad=8)
    ax.tick_params(colors='#cbd5e1', labelsize=11)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=160, facecolor=plt.gcf().get_facecolor(), edgecolor='none')
    plt.close()
    return buf.getvalue()


def generate_digits_cm_image():
    X, y = digits_dataset.data, digits_dataset.target
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test_scaled = digits_scaler.transform(X_test)
    y_pred = digits_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8.0, 6.2), facecolor='#0f172a')
    ax = plt.gca()
    ax.set_facecolor('#0f172a')
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        annot_kws={"size": 12, "weight": "bold"}
    )
    plt.title('10x10 Confusion Matrix Heatmap (Digits Dataset)', fontsize=14, fontweight='bold', color='#2dd4bf', pad=14)
    plt.xlabel('Predicted Label (Modelin Tahmini)', fontweight='bold', color='#cbd5e1', fontsize=12, labelpad=8)
    plt.ylabel('True Label (Gerçek Değer)', fontweight='bold', color='#cbd5e1', fontsize=12, labelpad=8)
    ax.tick_params(colors='#cbd5e1', labelsize=11)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=160, facecolor=plt.gcf().get_facecolor(), edgecolor='none')
    plt.close()
    return buf.getvalue()


@app.get("/visualizations/confusion-matrix/diabetes")
async def visual_diabetes_cm():
    png_data = generate_diabetes_cm_image()
    return Response(content=png_data, media_type="image/png")


@app.get("/visualizations/confusion-matrix/digits")
async def visual_digits_cm():
    png_data = generate_digits_cm_image()
    return Response(content=png_data, media_type="image/png")


def generate_diabetes_roc_image():
    csv_path = 'csvs/diabetes.csv' if os.path.exists('csvs/diabetes.csv') else 'diabetes.csv'
    df = pd.read_csv(csv_path, header=None)
    df.columns = DIABETES_COLUMNS + ['Outcome']
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test_scaled = diabetes_scaler.transform(X_test)
    y_prob = diabetes_model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(7.5, 5.5), facecolor='#0f172a')
    ax = plt.gca()
    ax.set_facecolor('#0f172a')
    plt.plot(fpr, tpr, color='#2dd4bf', lw=3.2, label=f'Logistic Regression (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='#94a3b8', lw=2.0, linestyle='--', label='Rastgele Tahmin (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Yanlış Pozitif Oranı)', color='#cbd5e1', fontweight='bold', fontsize=12, labelpad=8)
    plt.ylabel('True Positive Rate (Duyarlılık / Recall)', color='#cbd5e1', fontweight='bold', fontsize=12, labelpad=8)
    plt.title('Diyabet Modeli ROC Eğrisi (ROC-AUC)', color='#2dd4bf', fontweight='bold', fontsize=14, pad=14)
    plt.legend(loc='lower right', facecolor='#1e293b', edgecolor='#475569', labelcolor='#f8fafc', fontsize=11, framealpha=0.95, borderpad=0.8)
    ax.tick_params(colors='#cbd5e1', labelsize=11)
    ax.grid(True, linestyle=':', alpha=0.25, color='#475569')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=160, facecolor=plt.gcf().get_facecolor(), edgecolor='none')
    plt.close()
    return buf.getvalue()


@app.get("/visualizations/roc-curve/diabetes")
async def visual_diabetes_roc():
    png_data = generate_diabetes_roc_image()
    return Response(content=png_data, media_type="image/png")


def generate_elbow_image():
    k_range = list(range(1, 11))
    wcss = rfm_elbow_wcss

    plt.figure(figsize=(7.8, 5.6), facecolor='#0f172a')
    ax = plt.gca()
    ax.set_facecolor('#0f172a')
    plt.plot(k_range, wcss, marker='o', color='#2dd4bf', lw=3.0, markersize=9, markerfacecolor='#0d9488', markeredgecolor='#f0fdfa')

    # Annotate K=2 (Mathematical Elbow)
    plt.annotate('Matematiksel Dirsek (K=2)\n(En keskin varyans düşüşü)',
                 xy=(2, wcss[1]), xytext=(3.4, wcss[1] + 1800),
                 arrowprops=dict(facecolor='#f59e0b', shrink=0.08, width=2.0, headwidth=8),
                 color='#fbbf24', fontweight='bold', fontsize=10.5,
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#1e293b', edgecolor='#f59e0b', alpha=0.95))

    # Annotate K=4 (Commercial Optimal Selected)
    plt.annotate('Seçilen Model (K=4)\n(Ticari/Pazarlama Segmentasyonu)',
                 xy=(4, wcss[3]), xytext=(5.3, wcss[3] + 2200),
                 arrowprops=dict(facecolor='#38bdf8', shrink=0.08, width=2.0, headwidth=8),
                 color='#38bdf8', fontweight='bold', fontsize=10.5,
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#1e293b', edgecolor='#0284c7', alpha=0.95))

    plt.title('Elbow Yöntemi (Dirsek Grafiği - WCSS vs. K)', color='#2dd4bf', fontweight='bold', fontsize=14, pad=14)
    plt.xlabel('Küme Sayısı (K Değeri)', color='#cbd5e1', fontweight='bold', fontsize=12, labelpad=8)
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)', color='#cbd5e1', fontweight='bold', fontsize=12, labelpad=8)
    plt.xticks(k_range)
    ax.tick_params(colors='#cbd5e1', labelsize=11)
    ax.grid(True, linestyle=':', alpha=0.25, color='#475569')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=160, facecolor=plt.gcf().get_facecolor(), edgecolor='none')
    plt.close()
    return buf.getvalue()


@app.get("/visualizations/elbow-curve")
async def visual_elbow_curve():
    png_data = generate_elbow_image()
    return Response(content=png_data, media_type="image/png")


def generate_rfm_scatter_image(recency: float = None, frequency: float = None, monetary: float = None):
    plt.figure(figsize=(8.2, 6.0), facecolor='#0f172a')
    ax = plt.gca()
    ax.set_facecolor('#0f172a')

    cluster_config = {
        0: {'color': '#38bdf8', 'label': 'Ortalama / Düzenli (Küme 0)', 'alpha': 0.45, 'size': 26},
        1: {'color': '#f87171', 'label': 'Riskli / Kayıp (Küme 1)', 'alpha': 0.45, 'size': 26},
        2: {'color': '#34d399', 'label': 'Sadık / VIP (Küme 2)', 'alpha': 0.55, 'size': 32},
        3: {'color': '#fbbf24', 'label': 'Yeni Müşteri (Küme 3)', 'alpha': 0.45, 'size': 26},
    }

    if rfm_data is not None and not rfm_data.empty:
        for c_id, cfg in cluster_config.items():
            subset = rfm_data[rfm_data['Cluster'] == c_id]
            plt.scatter(
                subset['Recency'],
                subset['Monetary'],
                color=cfg['color'],
                alpha=cfg['alpha'],
                s=cfg['size'],
                edgecolors='none',
                label=cfg['label']
            )

    if recency is not None and monetary is not None:
        plt.scatter([recency], [monetary], color='#e11d48', s=260, marker='*', edgecolors='#ffffff', linewidth=2.0, zorder=15, label='[Sizin Müşteriniz]')
        y_offset = max(500, monetary * 0.1)
        plt.annotate(f'Sizin Müşteriniz\n(R={recency:.0f}, M={monetary:.0f}₺)',
                     xy=(recency, monetary), xytext=(recency + 25, monetary + y_offset),
                     arrowprops=dict(facecolor='#f43f5e', shrink=0.08, width=2.0, headwidth=8),
                     color='#fda4af', fontweight='bold', fontsize=10.5,
                     bbox=dict(boxstyle='round,pad=0.35', facecolor='#1e293b', edgecolor='#f43f5e', alpha=0.95))

    # Dinamik Y-ekseni: Aşırı uç değerlerin (whales) tüm grafiği 1 piksele sıkıştırmasını engellemek için
    # ve kullanıcının girdiği harcama tutarını daima kapsayacak şekilde optimize edilir
    y_max = max(12000, monetary * 1.15 if monetary is not None else 12000)
    plt.ylim(0, y_max)
    plt.xlim(-10, 390)

    plt.title('Gerçek Müşteri Segmentleri Dağılımı (Recency vs. Monetary)', color='#2dd4bf', fontweight='bold', fontsize=14, pad=14)
    plt.xlabel('Recency (Son Alışverişten Beri Geçen Gün)', color='#cbd5e1', fontweight='bold', fontsize=12, labelpad=8)
    plt.ylabel('Monetary (Toplam Harcama Tutarı ₺)', color='#cbd5e1', fontweight='bold', fontsize=12, labelpad=8)
    plt.legend(loc='upper right', facecolor='#1e293b', edgecolor='#475569', labelcolor='#f8fafc', fontsize=10.5, framealpha=0.95, borderpad=0.7)
    ax.tick_params(colors='#cbd5e1', labelsize=11)
    ax.grid(True, linestyle=':', alpha=0.25, color='#475569')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=160, facecolor=plt.gcf().get_facecolor(), edgecolor='none')
    plt.close()
    return buf.getvalue()


@app.get("/visualizations/cluster-scatter")
async def visual_cluster_scatter(recency: float = None, frequency: float = None, monetary: float = None):
    png_data = generate_rfm_scatter_image(recency, frequency, monetary)
    return Response(content=png_data, media_type="image/png")


def generate_arm_scatter_image():
    plt.figure(figsize=(7.8, 5.6), facecolor='#0f172a')
    ax = plt.gca()
    ax.set_facecolor('#0f172a')

    scatter = plt.scatter(
        ecommerce_rules['support'], ecommerce_rules['confidence'],
        c=ecommerce_rules['lift'], cmap='YlGnBu',
        s=ecommerce_rules['lift'] * 24, alpha=0.85, edgecolors='#334155', linewidth=0.9
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label('Lift (Kaldıraç Gücü)', color='#cbd5e1', fontweight='bold', fontsize=12, labelpad=8)
    cbar.ax.yaxis.set_tick_params(color='#cbd5e1')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#cbd5e1', fontsize=10.5)

    plt.title('Birliktelik Kuralları Dağılımı (Support vs Confidence)', color='#2dd4bf', fontweight='bold', fontsize=14, pad=14)
    plt.xlabel('Support (Destek Oranı)', color='#cbd5e1', fontweight='bold', fontsize=12, labelpad=8)
    plt.ylabel('Confidence (Güven Oranı)', color='#cbd5e1', fontweight='bold', fontsize=12, labelpad=8)
    ax.tick_params(colors='#cbd5e1', labelsize=11)
    ax.grid(True, linestyle=':', alpha=0.25, color='#475569')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=160, facecolor=plt.gcf().get_facecolor(), edgecolor='none')
    plt.close()
    return buf.getvalue()


@app.get("/visualizations/arm-scatter")
async def visual_arm_scatter():
    png_data = generate_arm_scatter_image()
    return Response(content=png_data, media_type="image/png")
