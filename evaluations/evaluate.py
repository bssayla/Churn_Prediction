from sklearn.metrics import (
    accuracy_score,
    f1_score, 
    precision_score, 
    recall_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.pipeline import Pipeline
import pandas as pd
from typing import Dict,Tuple


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series)-> Dict[str, float]:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'confusion': confusion
    }


def get_best_model(evaluations:Dict[str, Dict[str, float]],base_on:str)-> Tuple[str, str, float]:
    best_model = None
    best_score = 0
    best_pre_processing_function = None
    for pre_processing_function in evaluations:
        for model in evaluations[pre_processing_function]:
            if evaluations[pre_processing_function][model][base_on] > best_score:
                best_score = evaluations[pre_processing_function][model][base_on]
                best_model = model
                best_pre_processing_function = pre_processing_function
    return best_pre_processing_function, best_model, best_score

