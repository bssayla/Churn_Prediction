from evaluations.evaluate import *
from models.models import *
from data.data_scripts.data_preprocessing import (
    load_data,
    pre_processing_categorical_all_ordinal,
    drop_columns,
    split_data
)

import pytest



def test_evaluate_model():
    model = pipeline_model_RF()
    data = load_data('data/raw_data/data.csv')
    data = pre_processing_categorical_all_ordinal(data)
    data = drop_columns(data)
    X_train, X_test, y_train, y_test = split_data(data, my_test_size=0.2, my_random_state=42)
    model.fit(X_train, y_train)
    evaluation = evaluate_model(model, X_test, y_test)
    assert evaluation['accuracy'] > 0
    assert evaluation['f1'] > 0
    assert evaluation['precision'] > 0
    assert evaluation['recall'] > 0
    assert evaluation['roc_auc'] > 0
    assert evaluation['confusion'].shape == (2, 2)