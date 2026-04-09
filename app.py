import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from src.data_preprocessing import load_data, preprocess_data, split_data
from src.model_training import train_logistic_regression, train_random_forest, train_svm, train_xgboost, save_model
from src.evaluate_models import evaluate_model
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="AdOptima - Bank Loan Prediction", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = None
if 'target_encoder' not in st.session_state:
    st.session_state.target_encoder = None

def load_all_models():
    """Load all pre-trained models."""
    models_dir = 'models'
    models_dict = {}
    
    try:
        if os.path.exists(os.path.join(models_dir, 'logistic_regression.pkl')):
            models_dict['Logistic Regression'] = joblib.load(os.path.join(models_dir, 'logistic_regression.pkl'))
        
        if os.path.exists(os.path.join(models_dir, 'random_forest.pkl')):
            models_dict['Random Forest'] = joblib.load(os.path.join(models_dir, 'random_forest.pkl'))
        
        if os.path.exists(os.path.join(models_dir, 'svm.pkl')):
            models_dict['SVM'] = joblib.load(os.path.join(models_dir, 'svm.pkl'))
        
        if os.path.exists(os.path.join(models_dir, 'xgboost.pkl')):
            models_dict['XGBoost'] = joblib.load(os.path.join(models_dir, 'xgboost.pkl'))
        
        st.session_state.scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
        st.session_state.label_encoders = joblib.load(os.path.join(models_dir, 'label_encoders.pkl'))
        st.session_state.target_encoder = joblib.load(os.path.join(models_dir, 'target_encoder.pkl'))
        
        st.session_state.models = models_dict
        return True, models_dict
    except Exception as e:
        return False, str(e)

def train_all_models():
    """Train all models and save them."""
    try:
        if not os.path.exists('data/bank_fixed.csv'):
            st.error("Data file 'data/bank_fixed.csv' not found!")
            return False
        
        with st.spinner("Loading and preprocessing data..."):
            df = load_data('data/bank_fixed.csv')
            df_processed, label_encoders, scaler = preprocess_data(df, is_training=True)
            X_train, X_test, y_train, y_test = split_data(df_processed)
        
        with st.spinner("Training Logistic Regression..."):
            lr_model = train_logistic_regression(X_train, y_train)
            save_model(lr_model, 'logistic_regression.pkl')
        
        with st.spinner("Training Random Forest..."):
            rf_model = train_random_forest(X_train, y_train)
            save_model(rf_model, 'random_forest.pkl')
        
        with st.spinner("Training SVM..."):
            svm_model = train_svm(X_train, y_train)
            save_model(svm_model, 'svm.pkl')
        
        with st.spinner("Training XGBoost..."):
            xgb_model = train_xgboost(X_train, y_train)
            save_model(xgb_model, 'xgboost.pkl')
        
        # Load models back to session
        load_all_models()
        
        # Evaluate and display results
        st.session_state.training_results = {
            'X_test': X_test,
            'y_test': y_test,
            'models': {
                'Logistic Regression': lr_model,
                'Random Forest': rf_model,
                'SVM': svm_model,
                'XGBoost': xgb_model
            }
        }
        
        return True
    except Exception as e:
        st.error(f"Error during training: {str(e)}")
        return False

def prepare_input_for_prediction(input_data):
    """Prepare user input for prediction."""
    try:
        df = pd.DataFrame([input_data])
        df, _, _ = preprocess_data(df, is_training=False, 
                                   scaler=st.session_state.scaler,
                                   label_encoders=st.session_state.label_encoders)
        return df
    except Exception as e:
        st.error(f"Error preparing input: {str(e)}")
        return None

def make_predictions(input_df):
    """Make predictions with all models."""
    if input_df is None or len(st.session_state.models) == 0:
        st.error("Models not loaded. Please train models first.")
        return None
    
    predictions_dict = {}
    probabilities_dict = {}
    
    try:
        for model_name, model in st.session_state.models.items():
            pred = model.predict(input_df)[0]
            predictions_dict[model_name] = "Will Get Loan" if pred == 1 else "Will Not Get Loan"
            
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(input_df)[0]
                probabilities_dict[model_name] = {
                    'No': prob[0] * 100,
                    'Yes': prob[1] * 100
                }
            else:
                probabilities_dict[model_name] = None
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None
    
    return predictions_dict, probabilities_dict

# Main App
st.title("🏦 AdOptima - Bank Loan Prediction System")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Select Page", ["Home", "Train Models", "Make Predictions", "Model Comparison", "About"])

# Page: Home
if page == "Home":
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Welcome to AdOptima")
        st.write("""
        This application uses machine learning to predict whether a customer will subscribe to a term deposit.
        
        **Features:**
        - 🤖 4 Machine Learning Models (LR, RF, SVM, XGBoost)
        - 📊 Model Performance Comparison
        - 🔮 Multi-model Predictions
        - 📈 Real-time Training & Evaluation
        """)
    
    with col2:
        st.info("""
        **Quick Start:**
        1. Go to "Train Models" to train all models
        2. Use "Make Predictions" for new customer data
        3. Check "Model Comparison" for performance metrics
        """)
    
    st.markdown("---")
    
    # Check model status
    models_loaded, error_msg = load_all_models()
    
    if models_loaded:
        st.success(f"✓ All {len(st.session_state.models)} models loaded successfully!")
        col1, col2, col3, col4 = st.columns(4)
        for idx, model_name in enumerate(st.session_state.models.keys()):
            with st.columns(4)[idx]:
                st.metric("Model", model_name)
    else:
        st.warning(f"⚠️ Models not yet trained. Error: {error_msg}")

# Page: Train Models
elif page == "Train Models":
    st.header("🚀 Train All Models")
    
    st.info("""
    This page will train all 4 models on your dataset:
    - Logistic Regression
    - Random Forest
    - Support Vector Machine (SVM)
    - XGBoost
    """)
    
    if st.button("Train All Models", key="train_btn", use_container_width=True):
        if train_all_models():
            st.success("✓ All models trained successfully!")
            
            # Display training results
            if 'training_results' in st.session_state:
                st.header("Model Evaluation Results")
                
                col1, col2, col3, col4 = st.columns(4)
                cols = [col1, col2, col3, col4]
                
                for idx, (model_name, model) in enumerate(st.session_state.training_results['models'].items()):
                    metrics = evaluate_model(model, 
                                           st.session_state.training_results['X_test'],
                                           st.session_state.training_results['y_test'])
                    
                    with cols[idx]:
                        st.subheader(model_name)
                        st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                        st.metric("Precision", f"{metrics['Precision']:.4f}")
                        st.metric("Recall", f"{metrics['Recall']:.4f}")
                        st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
                        st.metric("ROC AUC", f"{metrics['ROC AUC']:.4f}")

# Page: Make Predictions
elif page == "Make Predictions":
    st.header("🔮 Make Predictions")
    
    models_loaded, _ = load_all_models()
    
    if not models_loaded:
        st.error("❌ Models not loaded! Please train models first.")
    else:
        st.write("Enter customer information to get predictions from all models:")
        
        col1, col2, col3 = st.columns(3)
        
        # Demographics
        with col1:
            st.subheader("Demographics")
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            job = st.selectbox("Job", ["admin.", "technician", "services", "management", "retired", 
                                       "blue-collar", "unemployed", "entrepreneur", "housemaid", "unknown", 
                                       "self-employed", "student"])
            marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
            education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
        
        # Contact Information
        with col2:
            st.subheader("Contact Info")
            default = st.selectbox("Has Credit in Default", ["yes", "no"])
            housing = st.selectbox("Has Housing Loan", ["yes", "no"])
            loan = st.selectbox("Has Personal Loan", ["yes", "no"])
            contact = st.selectbox("Contact Type", ["cellular", "telephone"])
        
        # Banking Activity
        with col3:
            st.subheader("Banking Activity")
            balance = st.number_input("Account Balance", value=0, step=100)
            day = st.number_input("Day of Month", min_value=1, max_value=31, value=15)
            month = st.selectbox("Month", ["jan", "feb", "mar", "apr", "may", "jun", 
                                          "jul", "aug", "sep", "oct", "nov", "dec"])
            duration = st.number_input("Duration (seconds)", value=180)
            campaign = st.number_input("Campaign Contacts", min_value=1, value=1)
            pdays = st.number_input("Days Since Last Contact", min_value=-1, value=-1)
            previous = st.number_input("Previous Campaign Contacts", min_value=0, value=0)
            poutcome = st.selectbox("Previous Campaign Outcome", ["unknown", "failure", "success"])
        
        # Prediction
        if st.button("Get Predictions from All Models", use_container_width=True):
            input_data = {
                'age': age,
                'job': job,
                'marital': marital,
                'education': education,
                'default': default,
                'balance': balance,
                'housing': housing,
                'loan': loan,
                'contact': contact,
                'day': day,
                'month': month,
                'duration': duration,
                'campaign': campaign,
                'pdays': pdays,
                'previous': previous,
                'poutcome': poutcome
            }
            
            input_df = prepare_input_for_prediction(input_data)
            
            if input_df is not None:
                predictions, probabilities = make_predictions(input_df)
                
                if predictions is not None:
                    st.header("Prediction Results")
                    
                    # Display predictions in columns
                    cols = st.columns(len(predictions))
                    for idx, (model_name, prediction) in enumerate(predictions.items()):
                        with cols[idx]:
                            st.subheader(model_name)
                            color = "green" if "Will" in prediction else "red"
                            st.markdown(f"**{prediction}**")
                            
                            if probabilities[model_name]:
                                prob_value = float(probabilities[model_name]['Yes'] / 100)
                                st.progress(prob_value, 
                                          text=f"Loan Probability: {probabilities[model_name]['Yes']:.2f}%")

# Page: Model Comparison
elif page == "Model Comparison":
    st.header("📊 Model Performance Comparison")
    
    models_loaded, _ = load_all_models()
    
    if not models_loaded or 'training_results' not in st.session_state:
        st.warning("⚠️ Please train models first to see comparison results.")
    else:
        # Evaluate all models
        metrics_dict = {}
        for model_name, model in st.session_state.training_results['models'].items():
            metrics = evaluate_model(model,
                                    st.session_state.training_results['X_test'],
                                    st.session_state.training_results['y_test'])
            metrics_dict[model_name] = metrics
        
        # Create comparison dataframe
        df_comparison = pd.DataFrame(metrics_dict).T
        st.dataframe(df_comparison.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
        
        # Visualizations
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Accuracy Comparison")
            fig = px.bar(df_comparison, y='Accuracy', title='Model Accuracy', 
                        color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("F1 Score Comparison")
            fig = px.bar(df_comparison, y='F1 Score', title='Model F1 Score',
                        color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            st.plotly_chart(fig, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Precision vs Recall")
            fig = go.Figure(data=[
                go.Scattergl(x=df_comparison['Precision'], y=df_comparison['Recall'],
                            mode='markers+text', text=df_comparison.index, textposition='top center',
                            marker=dict(size=10, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']))
            ])
            fig.update_layout(xaxis_title='Precision', yaxis_title='Recall')
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            st.subheader("ROC AUC Comparison")
            fig = px.bar(df_comparison, y='ROC AUC', title='Model ROC AUC',
                        color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            st.plotly_chart(fig, use_container_width=True)

# Page: About
elif page == "About":
    st.header("ℹ️ About AdOptima")
    
    st.write("""
    ## Project Overview
    AdOptima is a machine learning application designed to predict whether a customer will subscribe to a term deposit.
    
    ## Models Included
    1. **Logistic Regression** - Fast, interpretable linear classifier
    2. **Random Forest** - Ensemble method with multiple decision trees
    3. **Support Vector Machine (SVM)** - Powerful classifier for non-linear problems
    4. **XGBoost** - Advanced gradient boosting algorithm
    
    ## Dataset
    The project uses the Bank Marketing dataset which contains customer information and their response to marketing campaigns.
    
    ## Features
    - 15 input features capturing customer demographics and banking behavior
    - Binary classification (Will get loan / Will not get loan)
    - Multi-model ensemble approach for robust predictions
    
    ## Technologies Used
    - Python
    - Streamlit (Web Framework)
    - Scikit-learn (Machine Learning)
    - XGBoost (Advanced ML)
    - Plotly (Visualizations)
    """)
    
    st.markdown("---")
    st.info("📧 For questions or issues, please contact the development team.")
