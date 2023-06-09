# Predicting-Telco-Customer-Churn-and-findings-from-data-analysis-Using-machine-learning.
![dataimageddfe](https://github.com/justinjabo250/Time-Series-Forecasting-And-Analysis-Of-Store-Sales-Of-Corporation-Favorita-Products/assets/115732734/5bd480a2-f7de-46f6-9365-fda36d75c381)

# Introduction:

The Project research aims to estimate customer turnover and understand the demographic variations between churned and non-churned clients in order to analyze customer attrition at a telecoms company (Telco). Project Summary Customer turnover is a significant concern for telecom companies. Businesses can reduce customer turnover by developing predictive models and recognizing the factors that affect it.  Using machine learning techniques, we will compare the demographic characteristics of churned consumers to those who did not churn in this project.

Dataset The project makes use of the Telco Customer turnover dataset, which includes details on Telco clients’ demographics, services they subscribe to, and turnover rates. The data directory contains the dataset, which is provided in CSV format.

By examining previous data, seeing trends, and using other statistical techniques, we will use machine learning to identify which consumers are most likely to leave. This article will explore how to use customer data and behavioral traits to develop a classification model that can predict customer turnover using the CRISP-DM framework.

![dataimagesd](https://github.com/justinjabo250/Time-Series-Forecasting-And-Analysis-Of-Store-Sales-Of-Corporation-Favorita-Products/assets/115732734/9c00ea36-853a-4246-bb18-1d622a186e96)

# Plan Scenario:

In this project, we seek to determine the possibility that a client would leave the business, the primary churn indicators, as well as the retention tactics that may be used to avoid this issue.


# Project Description:

The amount of customers who discontinue doing business with a company during a specific time period is referred to as customer churn. In other terms, it refers to the frequency with which customers stop using a company’s goods or services. Churn can result from a number of circumstances, including unhappiness with the product or service, rival alternatives, modifications in consumer needs, or outside influences. Businesses need to understand and control customer turnover because it can significantly affect sales, expansion, and client happiness. In this project, we’ll determine the possibility that a client will leave the business, the important churn indicators, and the retention tactics that may be used to prevent this issue.

![datatssasas](https://github.com/justinjabo250/Time-Series-Forecasting-And-Analysis-Of-Store-Sales-Of-Corporation-Favorita-Products/assets/115732734/d65e9584-1623-4ea3-ba22-08959cbbc5e0)


# Objective:

To create a classification model that can reliably predict whether a customer would churn or not.

customer churn prediction using machine learning algorithms. For each model, evaluation measures (such accuracy, precision, recall, and F1-score). Comparison of the demographic makeup of churned and non-churned consumers. Visualizations, such as stacked bar charts, are used to display the findings.

![dsdsEsdasa](https://github.com/justinjabo250/Time-Series-Forecasting-And-Analysis-Of-Store-Sales-Of-Corporation-Favorita-Products/assets/115732734/76ed1882-b6ee-4f13-9cb5-9985e23e1660)

# Resources and Tools:

. **A dataset.**
. **Jupyter Notebook:** Scikit Learn, Pandas Profiling, Pandas, Matplolib, Seaborn, and other machine learning libraries are available.
Steps of the project
The project consists of the following sections:

# Data Reading
# Exploratory Data Analysis and Data Cleaning

. Data Visualization
. Feature Importance
. Feature Engineering
. Setting a baseline
. Splitting the data in training and testing sets
. Assessing multiple algorithms
. Algorithm selected: Gradient Boosting
. Hyperparameter tuning
. Performance of the model
. Drawing conclusions — Summary

![Dtataasa](https://github.com/justinjabo250/Time-Series-Forecasting-And-Analysis-Of-Store-Sales-Of-Corporation-Favorita-Products/assets/115732734/9ad05557-f976-418f-823b-af3453e97ad2)

# Exploratory Data Analysis (EDA):

Finding the pertinent features in your data can be one of the major challenges in developing a classification model that makes . . predictions. You may distinguish between churning and non-churning clients by identifying key features; this takes a thorough understanding of the business as well as significant data analysis to find patterns and trends in the data. Understanding data is aided by the strategy of posing queries and formulating hypotheses. To better comprehend the data, the following hypotheses and questions were developed.

# Hypothesis:

**Hypothesis Null:** Customers who have been with the company for a longer time are less likely to leave, Compared to clients who have been with the firm for a shorter time.

**Altenative:** The length of time a customer has been a customer of the business has no bearing on customer churn.

# Business Questions:

1. What is the overall churn rate for the company?
2. What are the demographics of customers who churned compared to those who did not?
3. How can the company reduce churn rate and retain more customers?

![dadaasdadasdas](https://github.com/justinjabo250/Time-Series-Forecasting-And-Analysis-Of-Store-Sales-Of-Corporation-Favorita-Products/assets/115732734/743d6e6f-f14f-452d-9261-dd068d1753d9)

Importing Libraries: 

. **from google.colab import drive
. import pandas as pd
. import numpy as np
. from matplotlib import pyplot as plt 
. import seaborn as sns
. from sklearn.impute import SimpleImputer
. from sklearn.preprocessing import OneHotEncoder
. from sklearn.preprocessing import LabelEncoder
. from sklearn.model_selection import train_test_split
. from sklearn.preprocessing import StandardScaler
. from imblearn.over_sampling import RandomOverSampler
. from sklearn.neighbors import KNeighborsClassifier
. from sklearn.metrics import confusion_matrix
. from sklearn.metrics import classification_report
. from sklearn.metrics import roc_curve
. from sklearn.metrics import roc_auc_score
. from sklearn.metrics import f1_score
. from sklearn.metrics import accuracy_score
. from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
. from sklearn.metrics import precision_recall_curve
. from sklearn.metrics import precision_recall_curve
. from sklearn.metrics import average_precision_score
. from sklearn.linear_model import LogisticRegression
. from sklearn.ensemble import GradientBoostingClassifier
. from sklearn.neural_network import MLPClassifier
. from sklearn.model_selection import GridSearchCV
. from sklearn.preprocessing import OrdinalEncoder
. from sklearn.pipeline import Pipeline
. import pickle**

. **import warnings
. warnings.filterwarnings("ignore")

. **%matplotlib inline****

![daatsSSS](https://github.com/justinjabo250/Time-Series-Forecasting-And-Analysis-Of-Store-Sales-Of-Corporation-Favorita-Products/assets/115732734/f092f684-dba9-4b46-a4d6-5097b1d9d378)

# Data Reading:
The first step of the analysis consists of reading and storing the data in a Pandas data frame using the pandas.read_csv function.

# Loading the dataset from google drive

drive.mount('/content/drive', force_remount=True)
customer = pd.read_csv('/content/drive/MyDrive/Azubi/lp3/Customer_Churn.csv')
Mounted at /content/drive
customer.head()
png

![DATSDADDFS](https://github.com/justinjabo250/Time-Series-Forecasting-And-Analysis-Of-Store-Sales-Of-Corporation-Favorita-Products/assets/115732734/aae5f2a7-8deb-4d6e-b066-40f8e777a2bb)

As was previously demonstrated, the data set consists of 19 independent variables that fall into one of three categories:

**1. Demographic Information:**

. **tenure:** The length of time a customer has been a customer of the business (multiple different number values).
. **client’s “gender”:** (Female, Male) Whether the client is a woman or a man.
. **SeniorCitizen:** Indicates whether the client is an older person (0, 1).
. **Partner:** Indicates whether or not the client is partnered (Yes, No).
. **Dependents:** Indicates whether the client is supported by others (Yes, No).

**2. Information about customer accounts:**

. **tenure:** The length of time a customer has been a customer of the business (multiple different number values).
. **Contract:** The type of existing contract for the customer (Month-to-Month, One-Year, Two-Year).
. **PaperlessBilling:** Whether the client uses paperless billing (Yes, No).
. **PaymentMethod:** The chosen payment method by the consumer (credit card, bank transfer, electronic check, paper check).
. **MontlyCharges:** The monthly charge made to the consumer (various numeric quantities).
. **TotalCharges** (many different numeric values): The total amount charged to the consumer.

**3. Services Information Details:**

. **Phone service:** If the client has a phone service, it is indicated by the words “Yes” or “No.”
. **MultipleLines:** Whether the customer has more than one line (no phone service, no service, yes service).
. **InternetServices:** Whether the client has a subscription to the company’s Internet service (DSL, Fiber optic, or No).
. **OnlineSecurity:** Indicates if the client has access to online security (Internet service available, No, Yes).
. **OnlineBackup:** Indicates whether or not the client has an online backup (Internet service unavailable, No, Yes).
. **DeviceProtection:** Indicates whether the client has device protection (Internet service not available, Not Available, Yes).
. **Tech support:** Whether the customer has access to tech help (no internet service, no, yes).
. **Streaming TV:** Whether the customer has access (no internet service, no, yes).
. **Streaming movies:** Indicates whether or not the client offers or has access streaming movies(no internet service, no, yes).


![dsdadtdasdsdsa](https://github.com/justinjabo250/Time-Series-Forecasting-And-Analysis-Of-Store-Sales-Of-Corporation-Favorita-Products/assets/115732734/bc36026a-345e-445e-9ac1-85bf2870b8e6)


2. **Exploratory Data Analysis and Data Cleaning**

Exploratory data analysis involves examining the key elements of a data set, typically using visualization techniques and summary statistics. Before conducting further analyses, the goal is to comprehend the data, find trends and anomalies, and test presumptions.

**Missing values and data types**

When EDA first started, we wanted to learn as much as we could about the data, which is why we used pandas.It’s useful to use the DataFrame.info method. This method outputs a brief summary of the data frame that includes the names of the columns and their data types, the number of non-null values, and the memory consumption.

![data-analyticsss](https://github.com/justinjabo250/Time-Series-Forecasting-And-Analysis-Of-Store-Sales-Of-Corporation-Favorita-Products/assets/115732734/5f4803e2-427b-4262-bb2c-2cbcc9beb1b1)

The data set has 21 columns and 7043 observations, as seen above. The data collection doesn’t appear to contain any null values, but we notice that the column TotalCharges was incorrectly identified as an object. This column is a numeric variable since it shows the total cost incurred by the customer. We need to convert this column into a numeric data type so that we may analyze it further. We may achieve this by use the pd.to_numeric function. When non-numeric data is encountered, this function raises an exception by default. However, we may use the errors=’coerce’ argument to bypass those cases and substitute them with a NaN.


![DstSsaSs](https://github.com/justinjabo250/Time-Series-Forecasting-And-Analysis-Of-Store-Sales-Of-Corporation-Favorita-Products/assets/115732734/308c4c40-05f5-4a1c-bd40-495d4123599e)

# Making judgments — Summary:

Using the Telco customer Churn dataset, we have gone through a whole end-to-end machine learning exercise in this post. We began by cleaning the data and using visualization to analyze it. After that, we feature engineered the categorical data into numeric variables in order to create a machine learning model. We experimented with six different machine learning algorithms using default parameters once the data had been transformed. Finally, we optimized the Gradient Boosting Classifier’s hyperparameters (best performance model) to achieve an accuracy of approximately 80% (about 6% better than the baseline).

It is crucial to emphasize that each project has a different set of precise machine learning task steps. Even though we followed a linear procedure in this post, machine learning projects typically follow an iterative approach rather than a linear one. As we learn more about the issue we’re trying to solve, earlier steps are frequently revisited.
