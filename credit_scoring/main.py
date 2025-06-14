import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os


data_path = r'C:\Users\budid\Downloads\credit dataset\german_credit_data.csv'
data = pd.read_csv(data_path)

print("Missing values before processing:")
print(data.isnull().sum())

numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)

if 'Unnamed: 0' in data.columns:
    data.drop(columns=['Unnamed: 0'], inplace=True)

data = pd.get_dummies(data, drop_first=True)

X = data.drop('Risk_good', axis=1) if 'Risk_good' in data.columns else data.iloc[:, :-1]
y = data['Risk_good'] if 'Risk_good' in data.columns else data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, os.path.join(model_dir, 'credit_model.pkl'))

print(f"\nModel saved to {os.path.join(model_dir, 'credit_model.pkl')}")
