import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load the Numerical Features (The "Input")
data_dict = np.load('../data/names_onehots.npy', allow_pickle=True).item()
X = data_dict['onehots']

if len(X.shape) == 3:
    X = X.reshape(X.shape[0], -1) 
    print(f"Reshaped X to 2D: {X.shape}")

# 2. Load the Labels (The "Answers")
train_df = pd.read_csv('../data/names_labels_train.csv', header=None, names=['name', 'label'])
y = train_df['label']     # This is the 0 or 1 column 

if len(X) != len(y):
    print(f"Warning: X has {len(X)} rows but y has {len(y)} rows. Trimming to match.")
    min_rows = min(len(X), len(y))
    X = X[:min_rows]
    y = y[:min_rows]

print("First few rows with correct headers:")
print(train_df.head())

# 3. Create and Train the Model
print("Training the AI model... this might take a minute.")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 4. Save the Model for your App
os.makedirs('../models', exist_ok=True)
joblib.dump(model, '../models/toxicity_model.pkl')

print("Success! Model trained and saved in the 'models' folder.")
