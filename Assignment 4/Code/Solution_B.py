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

X_train_das = pd.concat([X_train, X_val])
y_train_das = pd.concat([y_train, y_val])



##############################################################################################


# Version 1
print("Version 1 Parameters")
print("-"*50)
print("Logistic Regression: Penalty = L1 & Primal Formulation = False & Solver = liblinear")
print("Naive Bayes: Prior on Classes = (label1:0.5), (label2:0.5)")
print("Decision Tree: Max Depth=5\n")

clf1 = LogisticRegression(penalty='l1', solver='liblinear')
clf2 = GaussianNB([[0.5],[0.5]])
clf3 = DecisionTreeClassifier(max_depth=5)

clf_list = [clf1, clf2, clf3]

# Majority Voting
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

##############################################################################################

# Version 2
print("\n\nVersion 2 Parameters")
print("-"*50)
print("Logistic Regression: Penalty = L2 & Primal Formulation = False & Solver = lbfgs")
print("Naive Bayes: Prior on Classes = (label1:0.4), (label2:0.6)")
print("Decision Tree: Max Depth=10\n")

clf1 = LogisticRegression(penalty='l2')
clf2 = GaussianNB([[0.4],[0.6]])
clf3 = DecisionTreeClassifier(max_depth=10)

clf_list = [clf1, clf2, clf3]

# Majority Voting
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

##############################################################################################

# Version 3
print("\n\nVersion 3 Parameters")
print("-"*50)
print("Logistic Regression: Penalty = Elasticnet & Primal Formulation = False & Solver = SaGa")
print("Naive Bayes: Prior on Classes = (label1:0.6), (label2:0.4)")
print("Decision Tree: Max Depth=15\n")

clf1 = LogisticRegression(penalty='l2', solver='saga')
clf2 = GaussianNB([[0.6],[0.4]])
clf3 = DecisionTreeClassifier(max_depth=15)

clf_list = [clf1, clf2, clf3]

# Majority Voting
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

##############################################################################################

# Version 4
print("\n\nVersion 4 Parameters")
print("-"*50)
print("Logistic Regression: Penalty = None & Primal Formulation = False & Solver = lbfgs")
print("Naive Bayes: Prior on Classes = (label1:0.3), (label2:0.7)")
print("Decision Tree: Max Depth=25\n")

clf1 = LogisticRegression(penalty='none')
clf2 = GaussianNB([[0.3],[0.7]])
clf3 = DecisionTreeClassifier(max_depth=25)

clf_list = [clf1, clf2, clf3]

# Majority Voting
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

##############################################################################################

# Version 5
print("\n\nVersion 5 Parameters")
print("-"*50)
print("Logistic Regression: Penalty = L2 & Primal Formulation = True & Solver = liblinear")
print("Naive Bayes: Prior on Classes = (label1:0.7), (label2:0.3)")
print("Decision Tree: Max Depth=35\n")

clf1 = LogisticRegression(penalty='l2', dual=True, solver='liblinear')
clf2 = GaussianNB([[0.7],[0.3]])
clf3 = DecisionTreeClassifier(max_depth=35)

clf_list = [clf1, clf2, clf3]

# Majority Voting
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

##############################################################################################