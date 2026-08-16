import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

if not os.path.exists('models'): os.makedirs('models')

df = pd.read_csv("online_retail_II.csv")
df.dropna(subset=["Customer ID", "Invoice"], inplace=True)
df = df[~df["Invoice"].astype(str).str.contains("C")]
df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
df["TotalPrice"] = df["Quantity"] * df["Price"]
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
today_date = df["InvoiceDate"].max() + pd.Timedelta(days=2)

rfm = df.groupby("Customer ID").agg({"InvoiceDate": lambda d: (today_date - d.max()).days, "Invoice": "nunique", "TotalPrice": "sum"})
rfm.columns = ["Recency", "Frequency", "Monetary"]
rfm_log = np.log1p(rfm)
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)

kmeans = KMeans(n_clusters=4, init="k-means++", random_state=42, n_init=10)
kmeans.fit(rfm_scaled)

joblib.dump(kmeans, 'models/kmeans_model.pkl')
joblib.dump(scaler, 'models/rfm_scaler.pkl')
print("K-Means modeli kaydedildi.")
