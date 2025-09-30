import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import joblib
import os
from typing import Tuple, Dict, Any, Optional

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


class ChurnPredictor:
    """Handles all ML model training and prediction logic"""
    
    def __init__(self):
        self.preprocessor = None
        self.log_reg = None
        self.rf = None
        
    def create_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """Create preprocessing pipeline"""
        categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
        numerical_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
        
        numerical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        return ColumnTransformer([
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list]:
        """Preprocess the input dataframe"""
        df = df.copy()
        
        # Handle customer IDs
        if 'customerID' in df.columns:
            customer_ids = df['customerID'].tolist()
            df.drop('customerID', axis=1, inplace=True)
        else:
            customer_ids = list(range(len(df)))
        
        # Handle TotalCharges
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Handle target variable
        if 'Churn' in df.columns:
            df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
            y = df['Churn']
            X = df.drop('Churn', axis=1)
        else:
            y = None
            X = df
        
        return X, y, customer_ids
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train both models and return metrics"""
        # Create and fit preprocessor
        self.preprocessor = self.create_preprocessor(X)
        X_preprocessed = self.preprocessor.fit_transform(X)
        
        # Train Logistic Regression
        self.log_reg = LogisticRegression(
            max_iter=1000, 
            class_weight='balanced', 
            random_state=42
        )
        self.log_reg.fit(X_preprocessed, y)
        
        # Train Random Forest
        self.rf = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            class_weight='balanced', 
            random_state=42
        )
        self.rf.fit(X_preprocessed, y)
        
        # Save models
        self.save_models()
        
        return {"status": "Models trained successfully"}
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with both models"""
        if self.preprocessor is None:
            self.load_models()
        
        X_preprocessed = self.preprocessor.transform(X)
        
        log_probs = self.log_reg.predict_proba(X_preprocessed)[:, 1]
        rf_probs = self.rf.predict_proba(X_preprocessed)[:, 1]
        
        return log_probs, rf_probs
    
    def evaluate(self, y_true: pd.Series, log_probs: np.ndarray, rf_probs: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Evaluate model performance"""
        log_pred_class = (log_probs >= 0.5).astype(int)
        rf_pred_class = (rf_probs >= 0.5).astype(int)
        
        metrics = {
            "LogisticRegression": {
                "ROC_AUC": round(roc_auc_score(y_true, log_probs), 4),
                "Precision": round(precision_score(y_true, log_pred_class, zero_division=0), 4),
                "Recall": round(recall_score(y_true, log_pred_class, zero_division=0), 4),
                "F1": round(f1_score(y_true, log_pred_class, zero_division=0), 4),
            },
            "RandomForest": {
                "ROC_AUC": round(roc_auc_score(y_true, rf_probs), 4),
                "Precision": round(precision_score(y_true, rf_pred_class, zero_division=0), 4),
                "Recall": round(recall_score(y_true, rf_pred_class, zero_division=0), 4),
                "F1": round(f1_score(y_true, rf_pred_class, zero_division=0), 4),
            },
        }
        
        return metrics
    
    def save_models(self):
        """Save trained models to disk"""
        joblib.dump(self.log_reg, os.path.join(RESULTS_DIR, "log_reg.pkl"))
        joblib.dump(self.rf, os.path.join(RESULTS_DIR, "rf.pkl"))
        joblib.dump(self.preprocessor, os.path.join(RESULTS_DIR, "preprocessor.pkl"))
    
    def load_models(self):
        """Load trained models from disk"""
        self.log_reg = joblib.load(os.path.join(RESULTS_DIR, "log_reg.pkl"))
        self.rf = joblib.load(os.path.join(RESULTS_DIR, "rf.pkl"))
        self.preprocessor = joblib.load(os.path.join(RESULTS_DIR, "preprocessor.pkl"))
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """Complete pipeline: load, train, predict, evaluate"""
        try:
            df = pd.read_csv(file_path)
            X, y, customer_ids = self.preprocess_data(df)
            
            # Train models
            self.train_models(X, y)
            
            # Get predictions
            log_probs, rf_probs = self.predict(X)
            
            # Create result dataframe
            result_df = pd.DataFrame({
                'customerID': customer_ids,
                'LogisticRegressionProb': log_probs,
                'RandomForestProb': rf_probs,
            })
            
            if y is not None:
                result_df['ActualChurn'] = y
                metrics = self.evaluate(y, log_probs, rf_probs)
            else:
                metrics = None
            
            return {
                "success": True,
                "predictions": result_df.to_dict(orient='records'),
                "metrics": metrics
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }