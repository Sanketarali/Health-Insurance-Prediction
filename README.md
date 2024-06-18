
# Health-Insurance-Prediction
This project aims to predict the health insurance premiums of individuals based on their demographic, lifestyle, and health-related data.

# Prerequisites
To run this project, you will need the following:<br>

Python 3.x<br>
Jupyter Notebook<br>
scikit-learn library<br>
pandas library<br>
numpy library<br>

# How  did I do?
<h3>The dataset that I am using for the task of health insurance premium prediction is collected from Kaggle. It contains data about:<br></h3>

the age of the person<br>
gender of the person<br>
Body Mass Index of the person<br>
how many children the person is having<br>
whether the person smokes or not<br>
the region where the person lives<br>
and the charges of the insurance premium<br>

<h3>So let’s import the dataset and the necessary Python libraries that we need for this task:<br></h3>

import numpy as np<br>
import pandas as pd<br>
data = pd.read_csv("Health_insurance.csv")<br>
data.head()<br>

![image](https://github.com/Sanketarali/Health-Insurance-Prediction/assets/110754364/ad9575f1-f4ce-4a6a-9d17-e4da74bcae78)

<h3>Before moving forward, let’s have a look at whether this dataset contains any null values or not:<br></h3>

data.isnull().sum()<br>

![image](https://github.com/Sanketarali/Health-Insurance-Prediction/assets/110754364/6f44bb2b-ca4f-43f6-9acc-76cce158ca98)

 <h3>I will replace the values of the “sex” , “smoker” and "region" columns with 0 and 1 as both these columns contain string values:<br></h3>
 
 data["sex"] = data["sex"].map({"female": 0, "male": 1})<br>
data["smoker"] = data["smoker"].map({"no": 0, "yes": 1})<br>
data['region']=data['region'].map({'southwest':1,'southeast':2,
                   'northwest':3,'northeast':4})<br>
data.head()<br>

![image](https://github.com/Sanketarali/Health-Insurance-Prediction/assets/110754364/db610fdf-b296-4ce2-857d-000287d845ed)

# Health Insurance Premium Prediction Model
<h3>Now let’s move on to training a machine learning model for the task of predicting health insurance premiums. First, I’ll split the data into training and test sets:<br></h3>

x = np.array(data[["age", "sex", "bmi", "smoker"]])<br>
y = np.array(data["charges"])<br>

from sklearn.model_selection import train_test_split<br>
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)<br>

<h3>After using different machine learning algorithms, I found the random forest algorithm as the best performing algorithm for this task. So here I will train the model by using the random forest regression algorithm:<br></h3>

from sklearn.ensemble import RandomForestRegressor<br>
forest = RandomForestRegressor()<br>

<h3>Now let’s have a look at the predicted values of the model:<br></h3>

![image](https://github.com/Sanketarali/Health-Insurance-Prediction/assets/110754364/bb65bdae-f0e9-4b30-aa13-7c10604649c3)



