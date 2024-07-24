from models.models import *
from data.data_scripts.data_preprocessing import *
from evaluations.evaluate import evaluate_model
from typing import Dict
import joblib as jb


def train(data_raw_str: str) -> Dict[str, Dict[str, float]]:
    # Load the data
    #  data_raw_str = '../data/raw_data/data.csv'


    # Create the models
    first_model = pipeline_model_RF()
    second_model = pipeline_model_Ada()
    third_model = pipeline_model_SVC()


    pre_processing_type_list = [PreProcessingType.ONE_HOT, PreProcessingType.ORDINAL, PreProcessingType.DIFF]

    evaluations = {}

    for pre_processing_type in pre_processing_type_list:

        # Preprocess the data
        data = pre_processing(data_raw_str, pre_processing_type)
        
        # Split the data
        X_train, X_test, y_train, y_test = split_data(data)

        # Train the models
        first_model.fit(X_train, y_train)
        second_model.fit(X_train, y_train)
        third_model.fit(X_train, y_train)

        # Evaluate the models
        first_model_evaluation = evaluate_model(first_model, X_test, y_test)    
        second_model_evaluation = evaluate_model(second_model, X_test, y_test)
        third_model_evaluation = evaluate_model(third_model, X_test, y_test)

        evaluations[pre_processing_type._name_] = {
            'first_model': first_model_evaluation,
            'second_model': second_model_evaluation,
            'third_model': third_model_evaluation
        }
        # save model using joblib
        jb.dump(first_model, f'../models/trained_models/first_model_{pre_processing_type._name_}.pkl')
        jb.dump(second_model, f'../models/trained_models/second_model_{pre_processing_type._name_}.pkl')
        jb.dump(third_model, f'../models/trained_models/third_model_{pre_processing_type._name_}.pkl')

    return evaluations

    