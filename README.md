<div align="center">

# 🧠 AI-Driven Predictive & Analytical Intelligence Platform
### Yapay Zeka Destekli Tahmin, Sınıflandırma ve Müşteri Segmentasyon Paneli

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

<p align="center">
  Veri Ön İşleme, İkili/Çoklu Sınıflandırma, Müşteri Segmentasyonu ve Birliktelik Kuralı Madenciliği modellerini bir araya getiren <strong>uçtan uca etkileşimli makine öğrenmesi web uygulaması</strong>.
</p>

</div>

---

## 📌 Proje Hakkında (Overview)

Bu proje; makine öğrenmesi algoritmalarının veri ön işleme, model eğitimi, hiperparametre optimizasyonu ve performans değerlendirme aşamalarını **FastAPI** tabanlı modern bir mikroservis mimarisi ve **HTML5 / Tailwind CSS / Vanilla JS** etkileşimli ön yüzüyle birleştiren akademik ve endüstriyel bir karar destek panelidir.

Panel, veri biliminin 4 temel disiplinini canlı simülasyonlarla sunar:
1. **Veri Ön İşleme & Veri Sızıntısı Koruması (Data Preprocessing & Leakage Prevention)**
2. **Klinik Diyabet Risk Analizi (Binary Logistic Regression)**
3. **El Yazısı Rakam Tanıma (Multinomial Logistic Regression with Live Paint Board)**
4. **RFM Tabanlı Müşteri Segmentasyonu (Unsupervised K-Means++ Clustering)**
5. **Pazar Sepeti Ürün Öneri Motoru (Market Basket Analysis - Apriori Algorithm)**

---

## 🚀 Temel Modüller ve Özellikler

### 1. 🧪 Veri Ön İşleme Hattı (Pipeline Visualizer)
- **Eksik Veri Tespiti (Median Imputation):** Biyolojik değişkenlerdeki (Glikoz, Tansiyon, BMI) `0` değerleri tespit edilerek eğitim setinden hesaplanan medyanlar ile dinamik olarak doldurulur.
- **Standartlaştırma:** $Z$-skor dönüşümü (`StandardScaler`) ile tüm özellikler sıfır ortalama ve birim varyansa çekilir.
- **Veri Sızıntısı (Data Leakage) Koruması:** Ön işleme istatistikleri test setinden izole olarak yalnızca eğitim setinden türetilmiştir.

### 2. 🩺 Klinik Diyabet Risk Analizi (Binary Classification)
- **Model:** L2 regülarizasyonlu İkili Lojistik Regresyon (`LogisticRegression(random_state=42)`).
- **Performans:** **%71.43 Test Doğruluğu**, **0.8230 ROC-AUC Skoru**.
- **İnteraktif Özellik:** Form üzerinden girilen 8 klinik parametreye göre canlı **Hastalık Olasılığı Göstergesi (%0 - %100)** ve risk sınıflandırması.
- **Görselleştirmeler:**
  - 📊 *2x2 Confusion Matrix Heatmap*
  - 📈 *ROC Eğrisi (Receiver Operating Characteristic & AUC)*

### 3. ✍️ El Yazısı Rakam Tanıma (Multiclass Classification)
- **Model:** 10 Sınıflı Multinomial Lojistik Regresyon (`max_iter=1000, random_state=42`).
- **Veri Seti:** 1,797 örnekli 8x8 piksel `sklearn.datasets.load_digits`.
- **Performans:** **%97.22 Test Doğruluğu**, **0.97 F1-Score**.
- **İnteraktif Çizim Tahtası:**
  - **280x280 Dokunmatik Çizim Canvas'ı:** Doğrudan tarayıcıda fare veya dokunmatik ekranla çizim yapma.
  - **Direct Block Averaging & Aspect Ratio:** Çizilen piksel yoğunluklarını ve en-boy oranını bozmadan Scikit-Learn 8x8 piksel formatına indirgeyen doğrudan örnekleme algoritması.
  - **Canlı 8x8 Giriş Önizleme:** Modelin gördüğü 64 pikselin anlık önizleme penceresi.
  - **Rastgele / Belirli Veri Seti Örnek Yükleyici (0-9):** Gerçek veri seti örneklerini tek tıkla test etme.

### 4. 🛍️ Müşteri Segmentasyonu (RFM K-Means++ Clustering)
- **Model:** $K$-Means++ Kümeleme ($K=4$, $\text{log1p} + \text{StandardScaler}$).
- **Boyutlar:** 
  - **R (Recency):** Son alışverişten bu yana geçen gün.
  - **F (Frequency):** Toplam işlem / sipariş sayısı.
  - **M (Monetary):** Müşterinin bıraktığı toplam harcama tutarı.
- **🌟 Canlı Müşteri Konumlandırma Grafiği (Scatter Plot):** Kullanıcı yeni bir müşteri girdiğinde, arka plandaki 4 müşteri kümesi grafiği üzerinde müşterinin tam koordinatına parlayan bir kırmızı işaretçi yerleştirilir.
- **💡 Pazarlama Stratejisi Çıkarımı:** Küme bazlı aksiyon önerileri (*VIP Sadakat Kulübü*, *Riskli Müşteri Geri Kazanma İndirimi* vb.).
- **📉 Elbow Yöntemi (Dirsek Grafiği):** $K=2$ matematiksel bükülme noktası ve $K=4$ ticari segmentasyon seçiminin gerekçelendirmesi.

### 5. 🛒 Ürün Öneri Motoru (Market Basket - Apriori)
- **Model:** Apriori Birliktelik Kuralı Madenciliği ($\text{Support} \ge 0.02, \text{Lift} \ge 1.0$).
- **Veri:** Online Retail Fransa sipariş sepetleri.
- **Öneri Çıktısı:** Seçilen ürünle birlikte en sık satın alınan ürünler, **Lift (Kaldıraç Gücü)** ve **Confidence (Birlikte Alınma Güven Yüzdesi)** görsel ilerleme çubukları ile sunulur.
- **📊 Kural Dağılım Grafiği (Bubble Plot):** Destek, Güven ve Kaldıraç ilişkisini gösteren saçılım grafiği.

---

## 🛠️ Teknoloji Yığını (Tech Stack)

| Alan | Kullanılan Teknolojiler |
| :--- | :--- |
| **Backend & API** | Python 3.11+, FastAPI, Uvicorn, Pydantic |
| **Makine Öğrenmesi** | Scikit-Learn, Pandas, NumPy, Joblib, MLxtend |
| **Görselleştirme** | Matplotlib, Seaborn, HTML5 Canvas 2D |
| **Frontend UI** | HTML5, Modern CSS / Tailwind CSS (CDN), Vanilla JavaScript |

---

## 📂 Proje Dizin Yapısı

```
mpv/
├── main.py                     # FastAPI REST API, model endpoint'leri ve görselleştiriciler
├── index.html                  # Tek sayfa (SPA) interaktif dashboard arayüzü
├── requirements.txt            # Python bağımlılıkları
├── README.md                   # Proje dokümantasyonu
├── csvs/                       # Veri setleri (diabetes.csv, online_retail vb.)
│   └── diabetes.csv
├── models/                     # Eğitilmiş ve serileştirilmiş Scikit-Learn modelleri
│   ├── diabetes_model.pkl      # Logistic Regression (Diyabet)
│   ├── diabetes_scaler.pkl     # StandardScaler (Diyabet)
│   ├── diabetes_medians.pkl    # Eğitim seti medyan değerleri
│   ├── digits_model.pkl        # Multinomial Logistic Regression (Rakam)
│   ├── digits_scaler.pkl       # StandardScaler (Rakam)
│   ├── kmeans_model.pkl        # K-Means++ modeli (K=4)
│   ├── rfm_scaler.pkl          # RFM StandardScaler
│   ├── rfm_data.pkl            # Gerçek müşteri RFM verisi ve küme etiketleri (N=4,312)
│   ├── rfm_elbow_wcss.pkl      # Elbow WCSS değerleri (K=1..10)
│   └── ecommerce_rules_france.csv # Apriori kural tablosu
└── scripts/                    # Model eğitim ve hazırlık betikleri
    ├── diabetes_model_prep.py  # Diyabet modeli eğitim betiği
    ├── digits_model_prep.py    # Digits modeli eğitim betiği
    └── ecommerce_kmeans_prep.py # RFM K-Means & gerçek veri hazırlık betiği
```

---

## ⚡ Kurulum ve Çalıştırma

### 1. Depoyu Klonlayın
```bash
git clone <repo-url>
cd mpv
```

### 2. Sanal Ortamı Oluşturun ve Aktif Edin
```bash
python -m venv venv
source venv/bin/activate  # Windows için: venv\Scripts\activate
```

### 3. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### 4. Modelleri Hazırlayın (Opsiyonel / Zaten Kayıtlıdır)
```bash
python scripts/diabetes_model_prep.py
python scripts/digits_model_prep.py
python scripts/ecommerce_kmeans_prep.py
```

### 5. Backend Sunucusunu Başlatın
```bash
uvicorn main:app --reload --port 8000
```
> API servisi `http://127.0.0.1:8000` adresinde çalışmaya başlayacaktır.

### 6. Arayüzü Açın
- `index.html` dosyasını doğrudan herhangi bir modern web tarayıcısında (Chrome, Firefox, Edge, Safari) çift tıklayarak açabilirsiniz.

---

## 📡 API Uç Noktaları (Endpoints)

Otomatik interaktif Swagger dokümantasyonuna **`http://127.0.0.1:8000/docs`** adresinden erişebilirsiniz.

| Metot | Uç Nokta | Açıklama |
| :--- | :--- | :--- |
| `POST` | `/preprocess/diabetes` | Ham girdiyi medyan imputasyonu ve ölçekleme adımlarından geçirir. |
| `POST` | `/predict/diabetes` | Diyabet risk sınıflandırması ve olasılık skorunu hesaplar. |
| `POST` | `/predict/digit` | 64 piksellik diziden rakam sınıflandırması ve güven oranını üretir. |
| `GET` | `/digits/sample` | Gerçek veri setinden rastgele veya belirli bir rakam örneği döndürür. |
| `POST` | `/cluster/rfm` | R-F-M değerlerine göre müşteri segmentini ve küme numarasını döndürür. |
| `POST` | `/rules/recommend` | Seçilen ürün için en güçlü birliktelik önerilerini listeler. |
| `GET` | `/visualizations/confusion-matrix/diabetes` | Diyabet 2x2 Confusion Matrix ısı haritasını döndürür (PNG). |
| `GET` | `/visualizations/roc-curve/diabetes` | Diyabet ROC-AUC eğrisi grafiğini döndürür (PNG). |
| `GET` | `/visualizations/confusion-matrix/digits` | Rakam 10x10 Confusion Matrix ısı haritasını döndürür (PNG). |
| `GET` | `/visualizations/cluster-scatter` | Müşterinin koordinatını içeren canlı K-Means dağılım grafiğini döndürür (PNG). |
| `GET` | `/visualizations/elbow-curve` | K-Means WCSS Dirsek Eğrisi grafiğini döndürür (PNG). |
| `GET` | `/visualizations/arm-scatter` | Birliktelik kuralları Destek-Güven-Lift dağılım grafiğini döndürür (PNG). |

---

## 🎓 Akademik Referans ve Metodoloji

Bu proje, Dokuz Eylül Üniversitesi Bilgisayar Mühendisliği Bölümü bünyesinde gerçekleştirilen makine öğrenmesi araştırma ve staj raporlarındaki teorik ve deneysel metodolojilere dayanmaktadır.

- **Geliştirici:** Abdulkadir ANDIÇ
- **Akademik Danışman:** Prof. Dr. Alp KUT

---

<div align="center">
  <sub>Modern Veri Bilimi ve Yapay Zeka Uygulamaları © 2026</sub>
</div>
