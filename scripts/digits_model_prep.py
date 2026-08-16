import joblib
import os
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

if not os.path.exists('models'):
    os.makedirs('models')

# 1. Veri setini yukle (Rapordaki gibi: 64 ozellik, 10 sinif)
digits = load_digits()
X, y = digits.data, digits.target

# 2. Stratified Train-Test Split (Rapordaki gibi: test_size=0.2, random_state=42, stratify=y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. StandardScaler (Rapordaki gibi: fit_transform X_train uzerinde)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Multinomial Logistic Regression (Rapordaki gibi: random_state=42, max_iter=1000)
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# 5. Modelleri kaydet
joblib.dump(model, 'models/digits_model.pkl')
joblib.dump(scaler, 'models/digits_scaler.pkl')

print("Rapordaki egitim sekliyle birebir ayni Digits modeli ve scaler'i kaydedildi.")
print(f"Egitim dogrulugu: {model.score(X_train_scaled, y_train):.4f}")
print(f"Test dogrulugu: {model.score(X_test_scaled, y_test):.4f}")
