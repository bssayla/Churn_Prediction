from data.data_scripts.data_preprocessing import *
import pytest

@pytest.fixture
def data():
    data = load_data('data/raw_data/data.csv')
    return data
