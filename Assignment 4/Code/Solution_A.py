from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from Utility import *

df = pd.read_csv("ReplicatedAcousticFeatures-ParkinsonDatabase.csv")

df = df.loc[:, df.columns != 'ID']

# Split the Data into Training, Testing & Validation Dataset
train, val, test = train_test_val_split(df)

# Get the Features and Labels
X_train, y_train = train.loc[:, train.columns != 'Status'], train.loc[:, train.columns == 'Status']
X_val, y_val = val.loc[:, val.columns != 'Status'], val.loc[:, val.columns == 'Status']
X_test, y_test = test.loc[:, test.columns != 'Status'], test.loc[:, test.columns == 'Status']

# Classifiers
clf1 = LogisticRegression()
clf2 = GaussianNB()
clf3 = DecisionTreeClassifier()

clf_list = [clf1, clf2, clf3]

# Majority Voting
X_train_das = pd.concat([X_train, X_val])
y_train_das = pd.concat([y_train, y_val])

y_pred = majority_voting(X_train_das, y_train_das, X_test, y_test, clf_list)
acc_majority = accuracy_score(y_test, y_pred)


# Weighted Voting
y_pred = weighted_voting(X_train, y_train, X_val, y_val, X_test, y_test, clf_list)
acc_weighted = accuracy_score(y_test, y_pred)

# Individual Classifiers

# Logistic Regression
acc_logistic = get_result(X_train_das, y_train_das, X_test, y_test, clf1)

# Naive Bayes
acc_nb = get_result(X_train_das, y_train_das, X_test, y_test, clf2)

# Decision Trees
acc_dt = get_result(X_train_das, y_train_das, X_test, y_test, clf3)

# Print Results
print("Accuracy of Individual Classifiers:")
print("Logistic Regression: {}".format(acc_logistic))
print("Naive Bayes: {}".format(acc_nb))
print("Decision Trees: {}".format(acc_dt))

print("\n")

print("Accuracy of Ensembling Methods:")
print("Majority Voting: {}".format(acc_majority))
print("Weighted Voting: {}".format(acc_weighted))