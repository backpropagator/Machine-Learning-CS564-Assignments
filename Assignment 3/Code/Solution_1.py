from decision_tree import *
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

filename = 'heart.csv'
dataset = load_csv(filename)[1:]

# convert string attributes to integers
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)


max_depth = 3
max_split = 5
data_split = 0.7

print("Results using Decision Tree made from Scratch:\n")

# Using Gini Index
score = evaluate(dataset, decision_tree, data_split, max_depth, max_split, 'gini')
print("Accuracy using Gini Index: {}%".format(score))


# Using Information Gain
score = evaluate(dataset, decision_tree, data_split, max_depth, max_split, 'info_gain')
print("Accuracy using Information Gain: {}%".format(score))

print("\n")

# Using Sklearn Library
print("Results using Sklearn Library:\n")

# Pre-Processing
df = pd.read_csv("heart.csv")

columns = list(df.columns)

X = df[columns[0:len(columns)-1]]
y = df[columns[len(columns)-1]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=data_split)


#Decision Tree:
print("Descision Tree:")

# Using Gini Index
dtree = DecisionTreeClassifier(criterion="gini", max_depth=max_depth, min_samples_split=max_split)
dtree.fit(X_train, y_train)

y_predict = dtree.predict(X_test)
score = accuracy_score(y_test, y_predict)
print("Accuracy using Gini Index: {}%".format(score*100))

# Using Information Gain
dtree = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth, min_samples_split=max_split)
dtree.fit(X_train, y_train)

y_predict = dtree.predict(X_test)
score = accuracy_score(y_test, y_predict)
print("Accuracy using Information Gain: {}%".format(score*100))


#Logistic Regression:
print("\nLogistic Regression:")

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

y_predict = log_reg.predict(X_test)
score = accuracy_score(y_test, y_predict)
print("Accuracy using Logistic Regression: {}%".format(score*100))