import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import os

# Load datasets
train_df = pd.read_csv("data/Training.csv")
test_df = pd.read_csv("data/Testing.csv")

# Drop unnamed column if exists
train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
test_df = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]

# Separate features and target
X = train_df.drop(['prognosis'], axis=1)
y = train_df['prognosis']

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train model
model = DecisionTreeClassifier()
model.fit(X, y_encoded)

# Save model and label encoder
os.makedirs("models", exist_ok=True)
with open("models/model.pkl", "wb") as f:
    pickle.dump((model, le, X.columns.tolist()), f)

print("✅ Model training complete and saved to models/model.pkl")
