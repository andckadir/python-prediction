import joblib
import os
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

if not os.path.exists('models'): os.makedirs('models')

digits = load_digits()
X, y = digits.data, digits.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_scaled, y)

joblib.dump(model, 'models/digits_model.pkl')
joblib.dump(scaler, 'models/digits_scaler.pkl')
print("Digits modeli kaydedildi.")
