import streamlit as st
from typing import Dict
import joblib as jb
import pandas as pd


class C_Education_Level:
    HighSchool = 3
    Graduate = 2
    Uneducated = 5
    Unknown = 6
    College = 0
    PostGraduate = 4
    Doctorate = 1

def num_Education_Level(education_level: str) -> int:
    if education_level == "High School":
        return C_Education_Level.HighSchool
    elif education_level == "Graduate":
        return C_Education_Level.Graduate
    elif education_level == "Uneducated":
        return C_Education_Level.Uneducated
    elif education_level == "Unknown":
        return C_Education_Level.Unknown
    elif education_level == "College":
        return C_Education_Level.College
    elif education_level == "Post-Graduate":
        return C_Education_Level.PostGraduate
    elif education_level == "Doctorate":
        return C_Education_Level.Doctorate

def num_Income_Category(income_category: str) -> int:
    if income_category == "Less than $40K":
        return 4
    elif income_category == "$40K - $60K":
        return 1
    elif income_category == "$60K - $80K":
        return 2
    elif income_category == "$80K - $120K":
        return 3
    elif income_category == "$120K +":
        return 0
    elif income_category == "Unknown":
        return 5
    


def one_hot_marital_status(marital_status: str,customer_data):
    if marital_status == "Divorced":
        customer_data["Marital_Status_Divorced"] = 1
        customer_data["Marital_Status_Married"] = 0
        customer_data["Marital_Status_Single"] = 0
        customer_data["Marital_Status_Unknown"] = 0
    elif marital_status == "Married":
        customer_data["Marital_Status_Divorced"] = 0
        customer_data["Marital_Status_Married"] = 1
        customer_data["Marital_Status_Single"] = 0
        customer_data["Marital_Status_Unknown"] = 0
    elif marital_status == "Single":
        customer_data["Marital_Status_Divorced"] = 0
        customer_data["Marital_Status_Married"] = 0
        customer_data["Marital_Status_Single"] = 1
        customer_data["Marital_Status_Unknown"] = 0
    elif marital_status == "Unknown":
        customer_data["Marital_Status_Divorced"] = 0
        customer_data["Marital_Status_Married"] = 0
        customer_data["Marital_Status_Single"] = 0
        customer_data["Marital_Status_Unknown"] = 1
    
def one_hot_card_category(card_category: str,customer_data):
    if card_category == "Blue":
        customer_data["Card_Category_Blue"] = 1
        customer_data["Card_Category_Gold"] = 0
        customer_data["Card_Category_Platinum"] = 0
        customer_data["Card_Category_Silver"] = 0
    elif card_category == "Gold":
        customer_data["Card_Category_Blue"] = 0
        customer_data["Card_Category_Gold"] = 1
        customer_data["Card_Category_Platinum"] = 0
        customer_data["Card_Category_Silver"] = 0
    elif card_category == "Platinum":
        customer_data["Card_Category_Blue"] = 0
        customer_data["Card_Category_Gold"] = 0
        customer_data["Card_Category_Platinum"] = 1
        customer_data["Card_Category_Silver"] = 0
    elif card_category == "Silver":
        customer_data["Card_Category_Blue"] = 0
        customer_data["Card_Category_Gold"] = 0
        customer_data["Card_Category_Platinum"] = 0
        customer_data["Card_Category_Silver"] = 1

def one_hot_Gender(gender:str,customer_data):
    if gender == "F":
        customer_data["Gender_F"] = 1
        customer_data["Gender_M"] = 0
    elif gender == "M":
        customer_data["Gender_F"] = 0
        customer_data["Gender_M"] = 1

def get_customer_data()-> tuple:
    st.title("Predict Customer Churn")
    st.write("This app predicts the **Customer Churn** of a customer based on the dataset")
    st.write("Please fill in the details of the customer")
    customer_data = {}
    prediction = None


    Customer_Age = st.number_input("Customer Age", min_value=0, max_value=100, value=45)
    Gender = st.selectbox("Gender",("M","F"))
    Dependent_count = st.number_input("Dependent count", min_value=0, max_value=10, value=3)
    Education_Level = st.selectbox("Education Level",("High School","Graduate","Uneducated","Unknown","College","Post-Graduate","Doctorate"))
    Marital_Status = st.selectbox("Marital Status",("Married","Single","Unknown","Divorced"))
    Income_Category = st.selectbox("Income Category",("Less than $40K","$40K - $60K","$80K - $120K","$60K - $80K","Unknown","$120K +"),index=2)
    Card_Category = st.selectbox("Card Category",("Blue","Gold","Silver","Platinum"),index=0)
    Months_on_book = st.number_input("Months on book", min_value=0, max_value=100, value=39)
    Total_Relationship_Count = st.number_input("Total Relationship Count", min_value=0, max_value=10, value=5)
    Months_Inactive_12_mon = st.number_input("Months Inactive 12 mon", min_value=0, max_value=12, value=1)
    Contacts_Count_12_mon = st.number_input("Contacts Count 12 mon", min_value=0, max_value=12, value=3)
    Credit_Limit = st.number_input("Credit Limit", min_value=0, max_value=100000,value=12691)
    Total_Revolving_Bal = st.number_input("Total Revolving Bal", min_value=0, max_value=100000, value=777)
    Avg_Open_To_Buy = st.number_input("Avg Open To Buy", min_value=0, max_value=100000, value=11914)
    Total_Amt_Chng_Q4_Q1 = st.number_input("Total Amt Chng Q4 Q1", min_value=0.0, max_value=10.0, value=1.335)
    Total_Trans_Amt = st.number_input("Total Trans Amt", min_value=0, max_value=100000, value=1144)
    Total_Trans_Ct = st.number_input("Total Trans Ct", min_value=0, max_value=1000, value=42)
    Total_Ct_Chng_Q4_Q1 = st.number_input("Total Ct Chng Q4 Q1", min_value=0.0, max_value=10.0, value=1.625)
    Avg_Utilization_Ratio = st.number_input("Avg Utilization Ratio", min_value=0.0, max_value=1.0, value=0.061)

#       ['Customer_Age', 'Dependent_count', 'Education_Level', 'Income_Category',
#        'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive_12_mon',
#        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
#        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
#        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
#        'Marital_Status_Divorced', 'Marital_Status_Married',
#        'Marital_Status_Single', 'Marital_Status_Unknown', 'Card_Category_Blue',
#        'Card_Category_Gold', 'Card_Category_Platinum', 'Card_Category_Silver',
#        'Gender_F', 'Gender_M']
    if st.button("Predict"):
        customer_data["Customer_Age"] = Customer_Age
        customer_data["Dependent_count"] = Dependent_count
        customer_data["Education_Level"] = num_Education_Level(Education_Level)
        customer_data["Income_Category"] = num_Income_Category(Income_Category)
        customer_data["Months_on_book"] = Months_on_book
        customer_data["Total_Relationship_Count"] = Total_Relationship_Count
        customer_data["Months_Inactive_12_mon"] = Months_Inactive_12_mon
        customer_data["Contacts_Count_12_mon"] = Contacts_Count_12_mon
        customer_data["Credit_Limit"] = Credit_Limit
        customer_data["Total_Revolving_Bal"] = Total_Revolving_Bal
        customer_data["Avg_Open_To_Buy"] = Avg_Open_To_Buy
        customer_data["Total_Amt_Chng_Q4_Q1"] = Total_Amt_Chng_Q4_Q1
        customer_data["Total_Trans_Amt"] = Total_Trans_Amt
        customer_data["Total_Trans_Ct"] = Total_Trans_Ct
        customer_data["Total_Ct_Chng_Q4_Q1"] = Total_Ct_Chng_Q4_Q1
        customer_data["Avg_Utilization_Ratio"] = Avg_Utilization_Ratio
        one_hot_marital_status(Marital_Status,customer_data)
        one_hot_card_category(Card_Category,customer_data)
        one_hot_Gender(Gender,customer_data)
        st.write("Predicting Customer Churn: ")
        prediction = predict_customer_churn(customer_data)

    return customer_data, prediction



def predict_customer_churn(customer_data: dict):
    model = jb.load("models/trained_models/first_model_DIFF.pkl")
    customer_data = pd.DataFrame([customer_data])
    prediction = model.predict(customer_data)
    return prediction

def main():
    customer_data,prediction = get_customer_data()
    if prediction > .5:
        st.write("Customer is likely to stay")
    else:
        st.write("Customer is likely to churn")