import os
from src.data_preprocessing import load_data, preprocess_data, split_data
from src.model_training import train_logistic_regression, train_random_forest, train_svm, train_xgboost, save_model
from src.evaluate_models import evaluate_model, print_evaluation

def main():
    print("Loading and preprocessing data...")
    # Update the path to point to your dataset
    data_path = 'data/bank_fixed.csv'
    
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}. Please create a 'data' folder and place the dataset inside it.")
        return

    try:
        df = load_data(data_path)
    except Exception as e:
        print(f"Error loading the data: {e}")
        return

    df_processed, label_encoders, scaler = preprocess_data(df, is_training=True)
    X_train, X_test, y_train, y_test = split_data(df_processed)

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}\n")

    print("Training Logistic Regression...")
    lr_model = train_logistic_regression(X_train, y_train)
    save_model(lr_model, 'logistic_regression.pkl')
    
    print("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    save_model(rf_model, 'random_forest.pkl')

    print("Training SVM...")
    svm_model = train_svm(X_train, y_train)
    save_model(svm_model, 'svm.pkl')

    print("Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)
    save_model(xgb_model, 'xgboost.pkl')

    print("\n" + "="*50)
    print("EVALUATING MODELS")
    print("="*50 + "\n")
    
    lr_metrics = evaluate_model(lr_model, X_test, y_test)
    print_evaluation("Logistic Regression", lr_metrics)

    rf_metrics = evaluate_model(rf_model, X_test, y_test)
    print_evaluation("Random Forest", rf_metrics)

    svm_metrics = evaluate_model(svm_model, X_test, y_test)
    print_evaluation("SVM", svm_metrics)

    xgb_metrics = evaluate_model(xgb_model, X_test, y_test)
    print_evaluation("XGBoost", xgb_metrics)

if __name__ == "__main__":
    main()
