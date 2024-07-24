
"""
@Author: Ouaicha Mohamed
@email: Ouaicha47@gmail.com
@Description: This Model is going to do classification based on tha data we have to do churn prediction
@Date: 2024-07-17
@Data: {
        "CLIENTNUM",
        "Attrition_Flag", //categorical
        "Customer_Age",
        "Gender", //categorical
        "Dependent_count",
        "Education_Level", //categorical
        "Marital_Status", //categorical
        "Income_Category", //categorical
        "Card_Category", //categorical
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
        "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
        "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"
    }
"""
import pandas as pd
from enum import Enum
from typing import Tuple

class PreProcessingType(Enum):
    ONE_HOT = 0
    ORDINAL = 1
    DIFF = 2


def load_data(data_name_str: str)-> pd.DataFrame:
    return pd.read_csv(data_name_str)


def save_data(data:pd.DataFrame,save_path)-> None:
    data.to_csv(save_path, index=False)


def information_gain(X: pd.DataFrame, y: pd.Series)-> pd.Series:
    from sklearn.feature_selection import mutual_info_classif
    importance = mutual_info_classif(X, y)
    feature_importance = pd.Series(importance, X.columns)
    return feature_importance

def draw_feature_importance(feature_importance: pd.Series)-> None:
    import matplotlib.pyplot as plt
    feature_importance.plot(kind='barh', figsize=(10, 6))
    plt.show()

def pre_processing_categorical_ordinal(data: pd.DataFrame)-> pd.DataFrame:
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    data['Attrition_Flag'] = le.fit_transform(data['Attrition_Flag'])
    data['Education_Level'] = le.fit_transform(data['Education_Level'])
    data['Income_Category'] = le.fit_transform(data['Income_Category'])
    return data

def pre_processing_categorical_onehot(data: pd.DataFrame)-> pd.DataFrame:
    data = pd.get_dummies(data, columns=['Marital_Status', 'Card_Category','Gender'])
    return data

def pre_processing_categorical_all_ordinal(data: pd.DataFrame)-> pd.DataFrame:
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    data['Attrition_Flag'] = le.fit_transform(data['Attrition_Flag'])
    data['Education_Level'] = le.fit_transform(data['Education_Level'])
    data['Income_Category'] = le.fit_transform(data['Income_Category'])
    data['Marital_Status'] = le.fit_transform(data['Marital_Status'])
    data['Card_Category'] = le.fit_transform(data['Card_Category'])
    data['Gender'] = le.fit_transform(data['Gender'])
    return data
    
def pre_processing_categorical_all_onehot(data: pd.DataFrame)-> pd.DataFrame:
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    data = pd.get_dummies(data, columns=['Income_Category','Education_Level','Marital_Status', 'Card_Category','Gender'])
    data['Attrition_Flag'] = le.fit_transform(data['Attrition_Flag'])
    return data

def drop_columns(data: pd.DataFrame)-> pd.DataFrame:
    data.drop(
        [
            'CLIENTNUM',
            'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1', 
            'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
        ],
        axis=1, 
        inplace=True
        )
    return data


def features_selection(X:pd.DataFrame,y: pd.Series)-> pd.Series:
    feature_importance = information_gain(X, y)
    return feature_importance

def pre_processing(data_str: str,type: PreProcessingType = PreProcessingType.DIFF)-> pd.DataFrame:
    data = load_data(data_str)
    data = drop_columns(data)
    if type == PreProcessingType.ORDINAL:
        data = pre_processing_categorical_all_ordinal(data)

    elif type == PreProcessingType.ONE_HOT:
        data = pre_processing_categorical_all_onehot(data)

    elif type == PreProcessingType.DIFF:
        data = pre_processing_categorical_onehot(data)
        data = pre_processing_categorical_ordinal(data)

    else:
        raise ValueError("Invalid PreProcessingType")
    

    save_data(data,'../data/processed_data/data.csv')
    
    return data


def split_data(data: pd.DataFrame,my_test_size: float=0.2,my_random_state: int=42)-> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    from sklearn.model_selection import train_test_split
    X = data.drop('Attrition_Flag', axis=1)
    y = data['Attrition_Flag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=my_test_size, random_state=my_random_state)
    return X_train, X_test, y_train, y_test





# just for testing
if __name__=="__main__":
    print("start")
    data = load_data("data/raw_data/data.csv")
    save_data(data,'data/processed_data/data.csv')
    print("Done")