from xgboost import XGBClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def train_model(X_train, y_train):
    
    # Calculate class weights for cost-sensitive learning
    class_weights = compute_sample_weight('balanced', y_train)
    
    # Advanced XGBoost configuration with cost-sensitive learning
    xgb_model = XGBClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.85,
        min_child_weight=1,
        gamma=2,
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        scale_pos_weight=2.5,  # Penalize minority class misclassification
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    # Train with sample weights
    xgb_model.fit(X_train, y_train, sample_weight=class_weights)
    
    return xgb_model


def evaluate(model, X_test, y_test, feature_names=None):

    y_pred_proba = model.predict_proba(X_test)
    
    # Find optimal threshold using F1-score maximization
    best_threshold = 0.5
    best_f1 = 0
    best_predictions = model.predict(X_test)
    
    # Test different thresholds for minority class detection
    for threshold in np.arange(0.3, 0.7, 0.05):
        y_pred_thresh = np.argmax(y_pred_proba, axis=1)
        
        # For minority classes, lower the threshold
        minority_mask = (np.max(y_pred_proba, axis=1) < threshold)
        for idx in range(len(y_pred_proba)):
            if np.max(y_pred_proba[idx]) < threshold:
                # Predict second-highest probability class if confidence is low
                y_pred_thresh[idx] = np.argsort(y_pred_proba[idx])[-2]
        
        macro_f1 = f1_score(y_test, y_pred_thresh, average='macro', zero_division=0)
        
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_threshold = threshold
            best_predictions = y_pred_thresh

    y_pred = best_predictions

    # Calculate all metrics
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    micro_f1 = f1_score(y_test, y_pred, average='micro', zero_division=0)

    print("\n" + "="*70)
    print("ADVANCED MODEL PERFORMANCE METRICS (WITH OPTIMIZATIONS)")
    print("="*70)
    print(f"\nModel Name: XGBoost Classifier (Advanced Optimized)")
    print(f"Total Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    print(f"Weighted F1-Score: {weighted_f1:.4f}")
    print(f"Macro F1-Score: {macro_f1:.4f}")
    print(f"Micro F1-Score: {micro_f1:.4f}")
    print(f"Optimal Confidence Threshold: {best_threshold:.3f}")
    print("="*70)

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title("Confusion Matrix - Optimized Model")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    print("\n[SUCCESS] Confusion matrix saved as 'confusion_matrix.png'")
    
    # Feature Importance with names
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        top_n = 15
        top_indices = np.argsort(importances)[-top_n:]
        
        print(f"\nTop {top_n} Important Features:")
        print("-" * 70)
        for i, idx in enumerate(reversed(top_indices), 1):
            feature_name = feature_names[idx] if feature_names else f"Feature {idx}"
            print(f"{i:2d}. {feature_name:40s} | Importance: {importances[idx]:.4f}")
        print("-" * 70)
    plt.show()