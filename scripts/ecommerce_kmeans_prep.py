import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

if not os.path.exists('models'):
    os.makedirs('models')

csv_path = 'csvs/online_retail_II.csv' if os.path.exists('csvs/online_retail_II.csv') else 'online_retail_II.csv'
print(f"Loading dataset: {csv_path}")

df = pd.read_csv(csv_path)
df.dropna(subset=["Customer ID", "Invoice"], inplace=True)
df = df[~df["Invoice"].astype(str).str.contains("C")]
df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
df["TotalPrice"] = df["Quantity"] * df["Price"]
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
today_date = df["InvoiceDate"].max() + pd.Timedelta(days=2)

rfm = df.groupby("Customer ID").agg({
    "InvoiceDate": lambda d: (today_date - d.max()).days,
    "Invoice": "nunique",
    "TotalPrice": "sum"
})
rfm.columns = ["Recency", "Frequency", "Monetary"]

rfm_log = np.log1p(rfm)
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)

kmeans = KMeans(n_clusters=4, init="k-means++", random_state=42, n_init=10)
kmeans.fit(rfm_scaled)

# Add cluster labels to the rfm DataFrame
rfm["Cluster"] = kmeans.labels_

# Compute Elbow / WCSS (inertia) values for K=1 to 10
wcss_values = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    km.fit(rfm_scaled)
    wcss_values.append(round(km.inertia_, 2))

# Save models and artifacts
joblib.dump(kmeans, 'models/kmeans_model.pkl')
joblib.dump(scaler, 'models/rfm_scaler.pkl')
joblib.dump(rfm, 'models/rfm_data.pkl')
joblib.dump(wcss_values, 'models/rfm_elbow_wcss.pkl')

print("K-Means model, scaler, and rfm_data saved successfully.")
print(f"Total customers: {len(rfm)}")
print("Cluster distribution:\n", rfm['Cluster'].value_counts())

