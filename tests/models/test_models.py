from models.models import *
import pytest

def test_pipeline_model_RF():
    pipe = pipeline_model_RF()
    assert pipe.steps[0][0] == 'scaler'
    assert pipe.steps[1][0] == 'classifier'
    assert pipe.steps[1][1].__class__.__name__ == 'RandomForestClassifier'


def test_pipeline_model_Ada():
    pipe = pipeline_model_Ada()
    assert pipe.steps[0][0] == 'scaler'
    assert pipe.steps[1][0] == 'classifier'
    assert pipe.steps[1][1].__class__.__name__ == 'AdaBoostClassifier'


def test_pipeline_model_SVC():
    pipe = pipeline_model_SVC()
    assert pipe.steps[0][0] == 'scaler'
    assert pipe.steps[1][0] == 'classifier'
    assert pipe.steps[1][1].__class__.__name__ == 'SVC'

