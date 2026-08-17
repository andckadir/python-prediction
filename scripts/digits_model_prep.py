import joblib
import os
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

if not os.path.exists('models'):
    os.makedirs('models')

# 1. Load dataset (64 features, 10 classes: digits 0-9)
digits = load_digits()
X, y = digits.data, digits.target

# 2. Stratified Train-Test Split (test_size=0.2, random_state=42, stratify=y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Standardize features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Model training (Multinomial Logistic Regression)
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# 5. Save model and scaler
joblib.dump(model, 'models/digits_model.pkl')
joblib.dump(scaler, 'models/digits_scaler.pkl')

print("Digits model and scaler saved successfully.")
print(f"Train Accuracy: {model.score(X_train_scaled, y_train):.4f}")
print(f"Test Accuracy: {model.score(X_test_scaled, y_test):.4f}")
