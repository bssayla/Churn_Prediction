# Churn Prediction
This is a simple project to predict churn customers using a dataset from Kaggle. I used a 3 simple Models to predict the churn customers and compared them.


## Running Tests
To run tests, you can use the following commands
```bash
pytest
```
or 
```bash
make t # for non slow tests
make ta # for all tests
make ts # for slow tests
```

## DataSet
I got this dataset from [Credit Card customers](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers).
### Description (from Kaggle):
A manager at the bank is disturbed with more and more customers leaving their credit card services. They would really appreciate if one could predict for them who is gonna get churned so they can proactively go to the customer to provide them better services and turn customers' decisions in the opposite direction

Now, this dataset consists of 10,000 customers mentioning their age, salary, marital_status, credit card limit, credit card category, etc. There are nearly 18 features.

We have only 16.07% of customers who have churned. Thus, it's a bit difficult to train our model to predict churning customers.


### Note:
**PLEASE IGNORE THE LAST 2 COLUMNS (NAIVE BAYES CLAS…). I SUGGEST TO RATHER DELETE IT BEFORE DOING ANYTHING**

## Raport
You can find a detailed raport in the `notebooks` folder. It's a Jupyter Notebook file.
Where I did Data Analysis, Data Preprocessing,  Model Evaluation and Model Selection of the best model.
you can also find it as PDF or HTML if you want it

## Important
- Don't forget to install the requirements before running the tests or the code
```bash
pip install -r requirements.txt
```
- Use setup.py to install the package
```bash
pip install .
```
or
```bash
python setup.py install
```

## Run the Streamlit App
You can run the streamlit app by running the following command
```bash
streamlit run app.py
```
Interface will open in your browser wher you can put the data of the customer and get the prediction of the churn customer (will stay or will leave)
# Conclusion
- I worked on this project to get use to structuring my Machine Learning projects and not gather everything in one file or one notebook, and also use 'pytest','makefile','setup.py'... in my projects. Thats why i used some simple models just to practice.