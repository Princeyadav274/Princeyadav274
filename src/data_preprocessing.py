import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

def load_data(filepath):
    """Load the dataset."""
    return pd.read_csv(filepath, sep=';')

def preprocess_data(df, is_training=True, scaler=None, label_encoders=None):
    """Preprocess the dataset by encoding categorical and scaling numerical features."""
    df = df.copy()
    df = df.dropna()
    
    # Separate categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove target from categorical columns
    if 'y' in categorical_cols:
        categorical_cols.remove('y')
    if 'y' in numerical_cols:
        numerical_cols.remove('y')
    
    # Encode categorical columns
    if is_training:
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    else:
        # Use provided encoders for consistency
        for col in categorical_cols:
            if col in label_encoders:
                df[col] = label_encoders[col].transform(df[col].astype(str))
    
    # Scale numerical columns
    if is_training:
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        
        if not os.path.exists('models'):
            os.makedirs('models')
        joblib.dump(scaler, 'models/scaler.pkl')
        joblib.dump(label_encoders, 'models/label_encoders.pkl')
        
        # Target encoding
        if 'y' in df.columns:
            le_y = LabelEncoder()
            df['y'] = le_y.fit_transform(df['y'])
            joblib.dump(le_y, 'models/target_encoder.pkl')
    else:
        df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    return df, label_encoders, scaler

def split_data(df, target_col='y'):
    """Split the dataset into training and testing sets."""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=42)
