import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score, roc_auc_score)
from xgboost import XGBClassifier
import pickle
import os
from utils import load_and_preprocess
from config import RANDOM_STATE, TEST_SIZE, MODEL_PATH

def train_models(filepath="data/telco_churn.csv"):
    df = load_and_preprocess(filepath)
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, 
                                random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, 
                                random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(random_state=RANDOM_STATE, 
                                eval_metric='logloss')
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        results[name] = {
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1 Score": round(f1_score(y_test, y_pred), 4),
            "AUC-ROC": round(roc_auc_score(y_test, y_prob), 4)
        }
        trained_models[name] = model
    
    # Save best model (highest AUC-ROC)
    best_name = max(results, key=lambda x: results[x]['AUC-ROC'])
    best_model = trained_models[best_name]
    
    os.makedirs(MODEL_PATH, exist_ok=True)
    with open(f"{MODEL_PATH}best_model.pkl", 'wb') as f:
        pickle.dump(best_model, f)
    with open(f"{MODEL_PATH}feature_names.pkl", 'wb') as f:
        pickle.dump(list(X.columns), f)
    with open(f"{MODEL_PATH}X_test.pkl", 'wb') as f:
        pickle.dump(X_test, f)
    with open(f"{MODEL_PATH}y_test.pkl", 'wb') as f:
        pickle.dump(y_test, f)
        
    return results, best_name, trained_models, X_test, y_test, X

def load_best_model():
    with open(f"{MODEL_PATH}best_model.pkl", 'rb') as f:
        model = pickle.load(f)
    with open(f"{MODEL_PATH}feature_names.pkl", 'rb') as f:
        features = pickle.load(f)
    return model, features