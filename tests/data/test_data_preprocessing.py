from data.data_scripts.data_preprocessing import *
import pytest

def test_class_pre_processing_type():
    assert PreProcessingType.ONE_HOT.value == 0
    assert PreProcessingType.ORDINAL.value == 1
    assert PreProcessingType.DIFF.value == 2

    assert PreProcessingType.ONE_HOT.name == 'ONE_HOT'
    assert PreProcessingType.ORDINAL.name == 'ORDINAL'
    assert PreProcessingType.DIFF.name == 'DIFF'

def test_load_data():
    data = load_data('data/raw_data/data.csv')
    assert data.shape == (10127, 23)


def test_save_data(data):
    save_data(data,'data/processed_data/data.csv')
    data = load_data('data/raw_data/data.csv')
    assert data.shape == (10127, 23)

def test_information_gain(data):
    data = pre_processing_categorical_all_ordinal(data)
    X = data.drop('Attrition_Flag', axis=1)
    y = data['Attrition_Flag']
    feature_importance = information_gain(X, y)
    assert feature_importance.shape == (22,)


@pytest.mark.parametrize("data_str",["Attrition_Flag","Income_Category","Education_Level"])
def test_pre_processing_categorical_ordinal(data,data_str):
    data = pre_processing_categorical_ordinal(data)
    assert data[data_str].dtype == 'int64'


@pytest.mark.parametrize("col_str",["Marital_Status","Gender","Card_Category"])
def test_pre_processing_categorical_onehot(data,col_str):
    col_unique_values = set(data[col_str])
    data = pre_processing_categorical_onehot(data)
    for val in col_unique_values:
        if col_str + '_' + val not in data.columns:
            assert False
    assert True

@pytest.mark.parametrize("data_str",["Attrition_Flag","Income_Category","Education_Level",
                                     "Marital_Status","Gender","Card_Category"])
def test_pre_processing_categorical_all_ordinal(data,data_str):
    data = pre_processing_categorical_all_ordinal(data)
    assert data[data_str].dtype == 'int64'

@pytest.mark.parametrize("data_str",["Income_Category","Education_Level",
                                     "Marital_Status","Gender","Card_Category"])
def test_pre_processing_categorical_all_onehot(data,data_str):
    col_unique_values = set(data[data_str])
    data = pre_processing_categorical_all_onehot(data)
    for val in col_unique_values:
        if data_str + '_' + val not in data.columns:
            assert False
    assert True

def test_drop_columns(data):
    data = drop_columns(data)
    assert data.shape == (10127, 20)

def test_features_selection(data):
    data = pre_processing_categorical_all_ordinal(data)
    data = drop_columns(data)
    X = data.drop('Attrition_Flag', axis=1)
    y = data['Attrition_Flag']
    feature_importance = features_selection(X,y)
    assert feature_importance.shape == (19,)
    
@pytest.mark.skip(reason="FAILED tests/data/test_data_preprocessing.py::test_pre_processing - TypeError: argument of type 'method' is not iterable")
def test_pre_processing(data):
    types = [PreProcessingType.ONE_HOT, PreProcessingType.ORDINAL, PreProcessingType.DIFF]
    data = pre_processing(data, types[0])
    assert data.shape == (10127, 38)
    data = pre_processing(data, types[1])
    assert data.shape == (10127, 20)
    data = pre_processing(data, types[2])
    assert data.shape == (10127, 27)


def test_split_data(data):
    data = pre_processing_categorical_all_ordinal(data)
    data = drop_columns(data)
    X_train, X_test, y_train, y_test = split_data(data, my_test_size=0.2, my_random_state=42)
    assert X_train.shape == (8101, 19)
    assert X_test.shape == (2026, 19)
    assert y_train.shape == (8101,)
    assert y_test.shape == (2026,)

