# Term Deposit Prediction Project

## Overview
Banks need intelligent systems capable of forecasting the likelihood of a customer responding favorably to a marketing campaign. This project aims to develop a predictive system using machine learning that identifies the likelihood of a customer subscribing to a term deposit based on demographic, financial, and marketing variables.

This project implements 4 different machine learning models:
1. Logistic Regression
2. Random Forest
3. Support Vector Machine (SVM) (Optional, off by default for speed)
4. XGBoost Classifier

## Project Structure
- `requirements.txt` - Project dependencies.
- `src/`
  - `data_preprocessing.py` - Functions for cleaning and preprocessing the data (scaling numerical data, label encoding categorical data).
  - `model_training.py` - Functions for instantiating and training the ML models.
  - `evaluate_models.py` - Script with helper functions to evaluate the models' performance (Accuracy, Precision, Recall, F1, ROC-AUC).
- `main.py` - Primary script orchestrating the loading, preprocessing, model training, and evaluation workflows.
- `app.py` - Flask web application to serve the predictive insights as a web service.
- `templates/index.html` - HTML Frontend UI for the Flask Web app.

## How to setup and run
1. Ensure you have Python installed.
2. Install the necessary packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare the dataset:
   - Download the 'Bank Marketing' dataset from a repository like UCI Machine Learning Repository (e.g. `bank-additional-full.csv`).
   - Create a `data` folder inside this directory.
   - Place the CSV file into the `data/` folder.
4. Run the Machine Learning Pipeline:
   ```bash
   python main.py
   ```
   This will train the models and save them iteratively to a `models/` directory.
5. Run the web application:
   ```bash
   python app.py
   ```
6. Open your web browser and go to `http://127.0.0.1:5000/`.
