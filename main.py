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
# Load Pretrained Models, Scalers, and Statistics
# ---------------------------------------------------------------------------
# Diabetes (Binary Classification)
diabetes_model = joblib.load("models/diabetes_model.pkl")
diabetes_scaler = joblib.load("models/diabetes_scaler.pkl")

# Median values computed from training set (leakage-free)
if os.path.exists("models/diabetes_medians.pkl"):
    diabetes_medians = joblib.load("models/diabetes_medians.pkl")
else:
    diabetes_medians = {"Glucose": 117.0, "BloodPressure": 72.0, "SkinThickness": 29.0, "BMI": 32.3}

DIABETES_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
ZERO_AS_MISSING_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "BMI"]

# Clinical and statistical validity bounds (Winsorization / Outlier Capping)
DIABETES_BOUNDS = {
    "Pregnancies": {"min": 0.0, "max": 20.0, "label": "Hamilelik Sayısı", "unit": ""},
    "Glucose": {"min": 40.0, "max": 400.0, "label": "Glikoz", "unit": "mg/dL"},
    "BloodPressure": {"min": 30.0, "max": 240.0, "label": "Kan Basıncı", "unit": "mmHg"},
    "SkinThickness": {"min": 5.0, "max": 110.0, "label": "Cilt Kalınlığı", "unit": "mm"},
    "Insulin": {"min": 0.0, "max": 900.0, "label": "İnsülin", "unit": "mu U/ml"},
    "BMI": {"min": 10.0, "max": 75.0, "label": "Vücut Kitle İndeksi (BMI)", "unit": "kg/m²"},
    "DiabetesPedigreeFunction": {"min": 0.05, "max": 2.50, "label": "Soyağacı Fonk. (DPF)", "unit": ""},
    "Age": {"min": 18.0, "max": 120.0, "label": "Yaş", "unit": "yıl"},
}


def clean_and_impute_diabetes_features(raw_data: list[float]):
    """
    1. Accepts raw input feature list.
    2. Imputes biologically impossible zeros (Glucose, BP, Skin, BMI) with training medians.
    3. Caps invalid/extreme outlier values (e.g. Pregnancies < 0 or > 20, Age < 18 or > 120)
       to predefined clinical limits (Winsorization / Capping).
    4. Records all transformations and warning reasons.
    """
    raw_dict = dict(zip(DIABETES_COLUMNS, [float(v) for v in raw_data]))
    cleaned_dict = {}
    imputed_flags = {}
    clipped_flags = {}
    warnings = []

    for col in DIABETES_COLUMNS:
        val = raw_dict[col]
        bounds = DIABETES_BOUNDS[col]
        min_b = bounds["min"]
        max_b = bounds["max"]
        label = bounds["label"]

        # 1. Missing Value Detection and Median Imputation (biologically non-zero fields)
        if col in ZERO_AS_MISSING_COLUMNS and val <= 0:
            val = float(diabetes_medians[col])
            imputed_flags[col] = True
            warnings.append(f"{label}: Eksik veri (0) tespit edildi, eğitim medyanı ({val}) ile dolduruldu.")
        else:
            imputed_flags[col] = False

        # 2. Outlier and Extreme Value Capping (Winsorization / Clamping)
        orig_val = val
        if val < min_b:
            val = min_b
            clipped_flags[col] = {"from": orig_val, "to": min_b, "type": "min_clipped"}
            warnings.append(f"{label}: Girilen değer ({orig_val}) alt sınırın ({min_b}) altında olduğu için {min_b} değerine sabitlendi.")
        elif val > max_b:
            val = max_b
            clipped_flags[col] = {"from": orig_val, "to": max_b, "type": "max_clipped"}
            warnings.append(f"{label}: Girilen değer ({orig_val}) üst sınırın ({max_b}) üzerinde olduğu için {max_b} değerine sabitlendi.")
        else:
            clipped_flags[col] = None

        cleaned_dict[col] = round(val, 4)

    return raw_dict, cleaned_dict, imputed_flags, clipped_flags, warnings

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

# Cluster labels (Derived from centroid analysis)
CLUSTER_LABELS = {
    0: "Ortalama / Düzenli Müşteri",
    1: "Riskli / Kaybedilme Tehlikesi Olan Müşteri",
    2: "Sadık / Yüksek Değerli Müşteri (VIP)",
    3: "Yeni / Az Etkileşimli Müşteri",
}

# Association Rule Mining - Precomputed rule table
rules_path = "models/ecommerce_rules_france.csv" if os.path.exists("models/ecommerce_rules_france.csv") else "ecommerce_rules_france.csv"
ecommerce_rules = pd.read_csv(rules_path)


def _parse_frozenset_column(series: pd.Series) -> pd.Series:
    """Parses frozenset string representations (e.g. 'frozenset({\\'A\\', \\'B\\'})')
    from CSV into Python lists."""
    def parse(value: str):
        return re.findall(r"'([^']*)'", str(value))
    return series.apply(parse)


ecommerce_rules["antecedents_list"] = _parse_frozenset_column(ecommerce_rules["antecedents"])
ecommerce_rules["consequents_list"] = _parse_frozenset_column(ecommerce_rules["consequents"])


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class DiabetesInput(BaseModel):
    data: list[float]  # 8 features matching DIABETES_COLUMNS


class DigitInput(BaseModel):
    pixels: list[float]  # 64 pixel intensity values (8x8, 0-16 scale)


class RFMInput(BaseModel):
    recency: float
    frequency: float
    monetary: float


class ProductInput(BaseModel):
    product: str


# ---------------------------------------------------------------------------
# 1) CLASSIFICATION - Diabetes (Binary Logistic Regression)
# ---------------------------------------------------------------------------
@app.post("/predict/diabetes")
async def predict_diabetes(input: DiabetesInput):
    if len(input.data) != 8:
        raise HTTPException(status_code=400, detail="8 özellik bekleniyor (Insulin dahil).")
    
    raw_dict, cleaned_dict, imputed_flags, clipped_flags, warnings = clean_and_impute_diabetes_features(input.data)
    
    ordered_values = [cleaned_dict[c] for c in DIABETES_COLUMNS]
    input_df = pd.DataFrame([ordered_values], columns=DIABETES_COLUMNS)
    scaled = diabetes_scaler.transform(input_df)
    pred = diabetes_model.predict(scaled)
    prob = diabetes_model.predict_proba(scaled)[0][1]
    
    return {
        "prediction": int(pred[0]),
        "probability": float(prob) * 100,
        "raw": raw_dict,
        "cleaned": cleaned_dict,
        "imputed_flags": imputed_flags,
        "clipped_flags": clipped_flags,
        "warnings": warnings,
        "has_corrections": len(warnings) > 0,
    }


# ---------------------------------------------------------------------------
# 2) CLASSIFICATION - Handwritten Digits (Multinomial Logistic Regression)
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
# 3) CLUSTERING - RFM / K-Means++
# ---------------------------------------------------------------------------
@app.post("/cluster/rfm")
async def cluster_rfm(input: RFMInput):
    rfm_log = np.log1p([input.recency, input.frequency, input.monetary])
    input_df = pd.DataFrame([rfm_log], columns=["Recency", "Frequency", "Monetary"])
    scaled = rfm_scaler.transform(input_df)
    cluster = int(kmeans_model.predict(scaled)[0])
    return {"cluster": cluster, "label": CLUSTER_LABELS.get(cluster, f"Küme {cluster}")}


# ---------------------------------------------------------------------------
# 4) DATA PREPROCESSING - Live Pipeline Preview on Diabetes Data
# ---------------------------------------------------------------------------
@app.post("/preprocess/diabetes")
async def preprocess_diabetes(input: DiabetesInput):
    if len(input.data) != 8:
        raise HTTPException(status_code=400, detail="8 özellik bekleniyor (Insulin dahil).")

    raw_dict, cleaned_dict, imputed_flags, clipped_flags, warnings = clean_and_impute_diabetes_features(input.data)

    ordered_values = [cleaned_dict[c] for c in DIABETES_COLUMNS]
    input_df = pd.DataFrame([ordered_values], columns=DIABETES_COLUMNS)
    scaled_values = diabetes_scaler.transform(input_df)[0]
    scaled = dict(zip(DIABETES_COLUMNS, [round(float(v), 4) for v in scaled_values]))

    return {
        "raw": raw_dict,
        "imputed": cleaned_dict,
        "imputed_flags": imputed_flags,
        "clipped_flags": clipped_flags,
        "scaled": scaled,
        "warnings": warnings,
        "bounds": DIABETES_BOUNDS,
    }


# ---------------------------------------------------------------------------
# 5) ASSOCIATION RULE MINING - Market Basket (Online Retail / France)
# ---------------------------------------------------------------------------
@app.get("/rules/products")
async def list_products(limit: int = 60):
    """Returns unique products from antecedents and consequents for autocomplete."""
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
# 6) MODEL EVALUATION & VISUALIZATION
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

    # Annotate K=2 (Mathematical Elbow point)
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
    fig, ax = plt.subplots(figsize=(9.2, 5.6), facecolor='#0f172a')
    ax.set_facecolor('#0f172a')

    cluster_config = {
        0: {'color': '#38bdf8', 'label': 'Ortalama (K0)', 'alpha': 0.45, 'size': 26},
        1: {'color': '#f87171', 'label': 'Riskli/Kayıp (K1)', 'alpha': 0.45, 'size': 26},
        2: {'color': '#34d399', 'label': 'Sadık/VIP (K2)', 'alpha': 0.55, 'size': 32},
        3: {'color': '#fbbf24', 'label': 'Yeni Müşteri (K3)', 'alpha': 0.45, 'size': 26},
    }

    if rfm_data is not None and not rfm_data.empty:
        for c_id, cfg in cluster_config.items():
            subset = rfm_data[rfm_data['Cluster'] == c_id]
            ax.scatter(
                subset['Recency'],
                subset['Monetary'],
                color=cfg['color'],
                alpha=cfg['alpha'],
                s=cfg['size'],
                edgecolors='none',
                label=cfg['label']
            )

    # Dynamic limits:
    # 1. Dataset recency extends up to 375 days; ensure x_max is at least 420 so that all customers (>350 days)
    #    are clearly visible with ample margin and proper tick intervals reaching 400.
    # 2. If user enters recency > 350, expand x_max dynamically so user's star is always framed.
    x_max = max(420, (recency + 60) if recency is not None else 420)
    y_max = max(13500, (monetary * 1.22) if monetary is not None else 13500)

    has_user_point = recency is not None and monetary is not None

    if has_user_point:
        ax.scatter([recency], [monetary], color='#e11d48', s=320, marker='*', edgecolors='#ffffff', linewidth=2.2, zorder=15, label='Sizin Müşteriniz')
        
        # Adaptive annotation position:
        # Prevents clipping at edges and avoids overlapping any chart element
        if recency > x_max * 0.55:
            x_text = recency - (x_max * 0.06)
            ha_align = 'right'
        else:
            x_text = recency + (x_max * 0.06)
            ha_align = 'left'
            
        if monetary > y_max * 0.65:
            y_text = monetary - (y_max * 0.12)
        else:
            y_text = monetary + (y_max * 0.10)
            
        ax.annotate(
            f'Sizin Müşteriniz\n(R={recency:.0f}, M={monetary:.0f}₺)',
            xy=(recency, monetary),
            xytext=(x_text, y_text),
            ha=ha_align,
            arrowprops=dict(
                facecolor='#f43f5e',
                edgecolor='#ffffff',
                shrink=0.1,
                width=2.0,
                headwidth=7,
                headlength=6
            ),
            color='#fda4af',
            fontweight='bold',
            fontsize=10.5,
            bbox=dict(
                boxstyle='round,pad=0.45',
                facecolor='#1e293b',
                edgecolor='#f43f5e',
                linewidth=1.6,
                alpha=0.95
            ),
            zorder=20
        )

    ax.set_ylim(0, y_max)
    ax.set_xlim(-15, x_max)

    ax.set_title('Gerçek Müşteri Segmentleri Dağılımı (Recency vs. Monetary)', color='#2dd4bf', fontweight='bold', fontsize=13.5, pad=38)
    ax.set_xlabel('Recency (Son Alışverişten Beri Geçen Gün)', color='#cbd5e1', fontweight='bold', fontsize=11.5, labelpad=8)
    ax.set_ylabel('Monetary (Toplam Harcama Tutarı ₺)', color='#cbd5e1', fontweight='bold', fontsize=11.5, labelpad=8)
    
    # Legend positioned horizontally at the top to preserve maximum chart width
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 1.01),
        ncol=5 if has_user_point else 4,
        facecolor='#1e293b',
        edgecolor='#334155',
        labelcolor='#f8fafc',
        fontsize=9.5,
        framealpha=0.95,
        borderpad=0.5,
        handletextpad=0.4,
        columnspacing=1.0
    )
    
    ax.tick_params(colors='#cbd5e1', labelsize=10.5)
    ax.grid(True, linestyle=':', alpha=0.25, color='#475569')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=160, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig)
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
