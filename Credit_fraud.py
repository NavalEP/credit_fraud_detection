import datetime
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import LabelEncoder
train_data = pd.read_csv(r"C:\Users\naval\OneDrive\Desktop\Credit_card_fraud_detection\archive (3)\fraudTrain.csv")
train_data.info()
train_data.describe()
train_data.dtypes
train_data.columns

# 2- preprocessing 
train_data["trans_date_trans_time"] = pd.to_datetime(train_data["trans_date_trans_time"])
train_data["dob"] = pd.to_datetime(train_data["dob"])
train_data

train_data.drop(columns=['Unnamed: 0','cc_num','first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num','trans_date_trans_time'],inplace=True)
train_data

#Drop all rows that contain missing values 
train_data.dropna(ignore_index=True)
train_data

train_data.dropna(ignore_index=True)

encoder = LabelEncoder()
train_data["merchant"] = encoder.fit_transform(train_data["merchant"])
train_data["category"] = encoder.fit_transform(train_data["category"])
train_data["gender"] = encoder.fit_transform(train_data["gender"])
train_data["job"] = encoder.fit_transform(train_data["job"])

train_data

# 3-EDA

exit_counts = train_data["is_fraud"].value_counts()
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)  # Subplot for the pie chart
plt.pie(exit_counts, labels=["No", "YES"], autopct="%0.0f%%")
plt.title("is_fraud Counts")
plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()

# 4-Train the Model

X = train_data.drop(columns=["is_fraud"], inplace = False)
Y = train_data["is_fraud"]

model = SVC()
model.fit(X, Y)

model.score(X, Y)

test_data = pd.read_csv(r"C:\Users\naval\OneDrive\Desktop\Credit_card_fraud_detection\archive (3)\fraudTest.csv")
test_data

test_data.drop(columns=['Unnamed: 0','cc_num','first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num','trans_date_trans_time'],inplace=True)
test_data

encoder = LabelEncoder()
test_data["merchant"] = encoder.fit_transform(test_data["merchant"])
test_data["category"] = encoder.fit_transform(test_data["category"])
test_data["gender"] = encoder.fit_transform(test_data["gender"])
test_data["job"] = encoder.fit_transform(test_data["job"])

X_test = test_data.drop(columns=["is_fraud"], inplace = False)
Y_test = test_data["is_fraud"]

y_pred = model.predict(X_test)
y_pred

accuracy = accuracy_score(test_data['is_fraud'],y_pred)
accuracy

 