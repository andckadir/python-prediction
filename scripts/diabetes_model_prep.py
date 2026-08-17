import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

if not os.path.exists('models'):
    os.makedirs('models')

# 1. Load diabetes dataset (8 features, 1 target)
csv_path = 'csvs/diabetes.csv' if os.path.exists('csvs/diabetes.csv') else 'diabetes.csv'
df = pd.read_csv(csv_path, header=None)
df.columns = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]

X = df.drop(columns=['Outcome'])
y = df['Outcome']

# 2. Stratified train-test split (test_size=0.2, random_state=42, stratify=y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Standardize features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Model training (Binary Logistic Regression)
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Save model, scaler, and training medians
joblib.dump(model, 'models/diabetes_model.pkl')
joblib.dump(scaler, 'models/diabetes_scaler.pkl')

zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI']
temp = df[zero_cols].replace(0, np.nan)
medians = temp.median().to_dict()
joblib.dump(medians, 'models/diabetes_medians.pkl')

print("Diabetes model, scaler, and medians saved successfully.")
print(f"Train Accuracy: {model.score(X_train_scaled, y_train):.4f}")
print(f"Test Accuracy: {model.score(X_test_scaled, y_test):.4f}")
