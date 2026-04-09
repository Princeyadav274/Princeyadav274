import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib
import os

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression with optimized hyperparameters."""
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """Train Random Forest with optimized hyperparameters."""
    model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train):
    """Train SVM with optimized hyperparameters."""
    model = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        probability=True,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    """Train XGBoost with optimized hyperparameters."""
    model = XGBClassifier(
        n_estimators=150,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1])
    )
    model.fit(X_train, y_train)
    return model

def save_model(model, filename):
    """Save a trained model to the models directory."""
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(model, f'models/{filename}')
