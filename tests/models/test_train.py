from models.model_scripts.train import train
import pytest


@pytest.mark.slow
@pytest.mark.skip(reason="The Path is not correct")
def test_train():
    evaluations = train('data/raw_data/data.csv')

    assert type(evaluations) == dict
    assert evaluations.keys() == {'ORDINAL', 'ONE_HOT', 'DIFF'}
    
    assert evaluations['ORDINAL'].keys() == {'first_model', 'second_model', 'third_model'}
    assert evaluations['ORDINAL']['first_model'].keys() == {'accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'confusion'}
    assert evaluations['ORDINAL']['second_model'].keys() == {'accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'confusion'}
    assert evaluations['ORDINAL']['third_model'].keys() == {'accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'confusion'}
    
    assert evaluations['ONE_HOT'].keys() == {'first_model', 'second_model', 'third_model'}
    assert evaluations['ONE_HOT']['first_model'].keys() == {'accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'confusion'}
    assert evaluations['ONE_HOT']['second_model'].keys() == {'accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'confusion'}
    assert evaluations['ONE_HOT']['third_model'].keys() == {'accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'confusion'}
    
    assert evaluations['DIFF'].keys() == {'first_model', 'second_model', 'third_model'}
    assert evaluations['DIFF']['first_model'].keys() == {'accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'confusion'}
    assert evaluations['DIFF']['second_model'].keys() == {'accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'confusion'}
    assert evaluations['DIFF']['third_model'].keys() == {'accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'confusion'}
    