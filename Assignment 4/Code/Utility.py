from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import warnings

warnings.simplefilter("ignore")

def train_test_val_split(df, train_split=0.6, test_split=0.2, val_split=0.2):
	train, val, test = np.split(df.sample(frac=1, random_state=42), [int(train_split*len(df)), int((train_split+val_split)*len(df))])
	return train, val, test

def get_result(X_train, y_train, X_test, y_test, clf):
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	acc = accuracy_score(y_test, y_pred)

	return acc

def majority_voting(X_train, y_train, X_test, y_val, clf_list):
	predictions = []

	for clf in clf_list:
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)
		predictions.append(y_pred)

	y_pred = []

	for i in range(len(predictions[0])):
		pred = predictions[0][i] + predictions[1][i] + predictions[2][i]

		if pred >= 2:
			y_pred.append(1)
		else:
			y_pred.append(0)

	return np.array(y_pred)


def weighted_voting(X_train, y_train, X_val, y_val, X_test, y_test, clf_list):
	predictions = []
	weights = []

	for clf in clf_list:
		clf.fit(X_train, y_train)
		y_pred_val = clf.predict(X_val)

		weight = accuracy_score(y_test, y_pred_val)
		weights.append(weight)

	X_train_das = pd.concat([X_train, X_val])
	y_train_das = pd.concat([y_train, y_val])

	for clf in clf_list:
		clf.fit(X_train_das, y_train_das)
		y_pred = clf.predict(X_test)
		predictions.append(y_pred)

	y_pred = []

	for i in range(len(predictions[0])):
		pred = (weights[0]*predictions[0][i]) + (weights[1]*predictions[1][i]) + (weights[2]*predictions[2][i])

		pred = pred / sum(weights)

		if pred >= 0.5:
			y_pred.append(1)
		else:
			y_pred.append(0)

	return np.array(y_pred)