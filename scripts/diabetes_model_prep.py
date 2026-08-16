import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

if not os.path.exists('models'): os.makedirs('models')

columns = ['Hamilelik', 'Glikoz', 'Tansiyon', 'CiltKalinligi', 'Insulin', 'VucutKitleIndeksi', 'SoyagaciFonksiyonu', 'Yas', 'Sonuc']
df = pd.read_csv("diabetes.csv", names=columns)
df.drop("Insulin", axis=1, inplace=True)
non_zero_columns = ['Glikoz', 'Tansiyon', 'CiltKalinligi', 'VucutKitleIndeksi']
for col in non_zero_columns: df[col] = df[col].replace(0, np.nan).fillna(df[col].median())

y = df['Sonuc']
X = df.drop('Sonuc', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

joblib.dump(model, 'models/diabetes_model.pkl')
joblib.dump(scaler, 'models/diabetes_scaler.pkl')


columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI']

# Raporunuzdaki adımla birebir aynı: önce 0 -> NaN, SONRA median hesapla
temp = df[columns_with_zeros].replace(0, np.nan)
medians = temp.median().to_dict()

joblib.dump(medians, "models/diabetes_medians.pkl")
print(medians)


print("Diyabet modeli kaydedildi.")
