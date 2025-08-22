<H3>ENTER YOUR NAME: YASHWINI M</H3>
<H3>ENTER YOUR REGISTER NO. 212223230249</H3>
<H3>EX. NO.1</H3>
<H3>DATE: 22-08-2025</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```python
from google.colab import files
import pandas as pd
import io
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df=pd.read_csv('/content/Churn_Modelling.csv')
print(df)
print("\n")

x=df.iloc[:,:-1].values
print(x)
print("\n")

y=df.iloc[:,-1].values
print(y)
print("\n")

print(df.isnull().sum())
print("\n")

numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean().round(1))

print(df.isnull().sum())
print("\n")

y=df.iloc[:,-1].values
print(y)
print("\n")

df.duplicated()
print(df['EstimatedSalary'].describe())
print("\n")

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(df[numeric_cols])) # Scale only numeric columns
print(df1)
print("\n")


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train)
print(len(x_train))
print("\n")
print(x_test)
print(len(x_test))
```

## OUTPUT:
<img width="770" height="249" alt="image" src="https://github.com/user-attachments/assets/dc01965d-1d60-412a-a7e2-e7a42fdda359" />
<img width="666" height="541" alt="image" src="https://github.com/user-attachments/assets/c6c7bff5-72e1-4596-b83e-1bfb55461055" />
<img width="630" height="550" alt="image" src="https://github.com/user-attachments/assets/2a38e6a0-b217-46b3-90e3-b5714a301daf" />
<img width="845" height="470" alt="image" src="https://github.com/user-attachments/assets/55ce1682-2e21-40d0-aa33-7da03540d0c7" />
<img width="627" height="678" alt="image" src="https://github.com/user-attachments/assets/37ce787f-6254-4f8e-940d-a2036606afb3" />



## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


