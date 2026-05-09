import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
import category_encoders as ce
from imblearn.over_sampling import SMOTE

def preprocess(df):

    # Remove rows with NaN values
    df = df.dropna()

    # Target
    y = df['Accident_severity']
    X = df.drop('Accident_severity', axis=1)

    le = LabelEncoder()
    y = le.fit_transform(y)

    # High-cardinality columns for target encoding
    high_card_cols = [
        'Weather_conditions',
        'Road_surface_type',
        'Type_of_vehicle',
        'Road_Weather',
        'Vehicle_Road',
        'Time_Weather',
        'Driver_Age_Experience',
        'Junction_Road_Type'
    ]
    
    # Identify all categorical columns
    categorical_cols = X.select_dtypes(include='object').columns.tolist()
    low_card_cols = [col for col in categorical_cols if col not in high_card_cols]

    # Encode low-cardinality categorical columns with LabelEncoder
    X_encoded = X.copy()
    label_encoders = {}
    for col in low_card_cols:
        le_col = LabelEncoder()
        X_encoded[col] = le_col.fit_transform(X_encoded[col].astype(str))
        label_encoders[col] = le_col

    # Target Encoding with KFold for high-cardinality columns
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Initialize columns for encoding
    for col in high_card_cols:
        if col in X_encoded.columns:
            X_encoded[col] = 0.0

    for train_idx, val_idx in kf.split(X, y):
        X_train_fold = X.iloc[train_idx].copy()
        X_val_fold = X.iloc[val_idx].copy()
        y_train_fold = y[train_idx]

        encoder = ce.TargetEncoder(cols=high_card_cols)
        encoder.fit(X_train_fold, y_train_fold)

        encoded_vals = encoder.transform(X_val_fold)
        
        # Update X_encoded with encoded values for validation indices
        for col in high_card_cols:
            if col in encoded_vals.columns:
                mask = pd.Series(False, index=X_encoded.index)
                mask.iloc[val_idx] = True
                X_encoded.loc[mask, col] = encoded_vals[col].values

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = X_encoded.copy()
    numeric_cols = X_scaled.select_dtypes(include=[np.number]).columns.tolist()
    X_scaled[numeric_cols] = scaler.fit_transform(X_scaled[numeric_cols])

    # Aggressive SMOTE for minority class oversampling
    smote = SMOTE(
        random_state=42,
        k_neighbors=3,  # More aggressive
        sampling_strategy={0: 200, 1: 300}  # Significantly oversample minorities
    )
    
    try:
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    except:
        # Fallback if aggressive SMOTE fails
        smote = SMOTE(
            random_state=42,
            k_neighbors=3,
            sampling_strategy='not majority'
        )
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    return X_resampled, y_resampled, le, scaler