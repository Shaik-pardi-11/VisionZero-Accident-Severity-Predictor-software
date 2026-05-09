import pandas as pd
from sklearn.model_selection import train_test_split

from feature_engineering import feature_engineering
from preprocess import preprocess
from model import train_model, evaluate

# Load data
df = pd.read_csv("../data/Road.csv")

# Step 1: Feature Engineering
df = feature_engineering(df)

# Step 2: Preprocessing + Target Encoding + Aggressive SMOTE
X, y, le, scaler = preprocess(df)

# Store feature names before train-test split
feature_names = X.columns.tolist()

# Step 3: Train-Test Split (on already resampled data)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 4: Train Model
model = train_model(X_train, y_train)

# Step 5: Evaluate with threshold tuning and feature names
evaluate(model, X_test, y_test, feature_names)