from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC


def pipeline_model_RF()-> Pipeline:
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier())
    ])
    return pipe


def pipeline_model_Ada()-> Pipeline:
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', AdaBoostClassifier())
    ])
    return pipe

def pipeline_model_SVC()-> Pipeline:
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC())
    ])
    return pipe

    