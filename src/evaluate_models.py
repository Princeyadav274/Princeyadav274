from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    
    # Check if the model has predict_proba
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, probs)
    else:
        # Fallback if no probabilities (some instances depending on model configuration)
        roc_auc = roc_auc_score(y_test, predictions)
    
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    }

def print_evaluation(name, metrics):
    print(f"--- {name} ---")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("\n")
