
# coding: utf-8

# In[61]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

url = "https://storage.googleapis.com/py_ds_basic/kaggle_titanic_train.csv"
titanic_train = pd.read_csv(url)
age_median = np.nanmedian(titanic_test.Age)
new_Age = np.where(titanic_train.Age.isnull(), age_median, titanic_train.Age)
titanic_train.Age = new_Age
# print(titanic_train.describe())
# print(titanic_train.isnull())
# plt.hist(titanic_train.Age)
# plt.show()
# print(titanic_train.Embarked.value_counts())
new_Embarked = np.where(titanic_train.Embarked.isnull(), "S", titanic_train.Embarked)
titanic_train.Embarked = new_Embarked
fare_median = np. nanmedian(titanic_train.Fare)
new_Fare = np.where(titanic_train.Fare.isnull(), fare_median, titanic_train.Fare)
titanic_train.Age = new_Age
# print(titanic_train.Embarked.isnull().sum())
# print(titanic_train.describe())
label_encoder =LabelEncoder()
encoded_Sex = label_encoder.fit_transform(titanic_train.Sex)
titanic_train.Sex = encoded_Sex
encoded_Embarked = label_encoder.fit_transform(titanic_train.Embarked)
titanic_train.Embarked = encoded_Embarked
#print(titanic_train.head())
titanic_x = pd.DataFrame([titanic_train.Pclass,
                          encoded_Sex,
                          new_Age,
                          new_Fare,
                          encoded_Embarked
]).T
titanic_y = titanic_train.Survived
# print(titanic_x.head())
# print(titanic_y.head())
# print("===")
from sklearn.cross_validation import train_test_split
from sklearn import metrics, tree
train_x, test_x, train_y, test_y = train_test_split(titanic_x,titanic_y, test_size = 0.3)
decision_clf = tree.DecisionTreeClassifier(random_state= 87)
decision_clf.fit(train_x,train_y)

test_y_predicted = decision_clf.predict(test_x)

accuracy = metrics.accuracy_score(test_y,test_y_predicted)
print(accuracy)
#print(titanic_test.isnull().sum())


url = "https://storage.googleapis.com/py_ds_basic/kaggle_titanic_test.csv"
to_submit = pd.read_csv(url)
age_median = np.nanmedian(to_submit.Age)
new_Age = np.where(to_submit.Age.isnull(), age_median, to_submit.Age)
to_submit.Age = new_Age
# print(titanic_test.describe())
# print(titanic_test.isnull())
# plt.hist(titanic_test.Age)
# plt.show()
# print(titanic_test.Embarked.value_counts())
new_Embarked = np.where(to_submit.Embarked.isnull(), "S", to_submit.Embarked)
to_submit.Embarked = new_Embarked
fare_median = np. nanmedian(to_submit.Fare)
new_Fare = np.where(to_submit.Fare.isnull(), fare_median, to_submit.Fare)
to_submit.Age = new_Age
# print(titanic_test.Embarked.isnull().sum())
# print(titanic_test.describe())
label_encoder =LabelEncoder()
encoded_Sex = label_encoder.fit_transform(to_submit.Sex)
to_submit.Sex = encoded_Sex
encoded_Embarked = label_encoder.fit_transform(to_submit.Embarked)
to_submit.Embarked = encoded_Embarked
#print(to_submit.head())
to_submit_x = pd.DataFrame([to_submit.Pclass,
                          encoded_Sex,
                          new_Age,
                          new_Fare,
                          encoded_Embarked
]).T

to_submit_y = decision_clf.predict(to_submit_x)
print(to_submit_y[0:5])

to_submit_dict ={
    "PassengerID": to_submit["PassengerId"],
    "Survived": to_submit_y
}

to_submit_df = pd.DataFrame(to_submit_dict)

to_submit_df.to_csv("to_submit.csv" , index = False)


# In[ ]:




# In[ ]:



