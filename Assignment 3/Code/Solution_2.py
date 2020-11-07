from decision_tree import *
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

filename = 'heart.csv'
dataset = load_csv(filename)[1:]

max_depth = 3
max_split = 5

# convert string attributes to integers
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)


print("Results using Gini Index:\n")
# 1st part (Variation with Train-Test Split)
split = []
acc = []

print("Split(%) \t Accuracy(%)")

for i in range(50,100,5):
	score = evaluate(dataset, decision_tree, i/100, max_depth, max_split, 'gini')
	print("{} \t {}%".format(i,score))
	split.append(i)
	acc.append(score)

plt.plot(split,acc)
plt.xlabel("Split(%)")
plt.ylabel("Test Accuracy(%)")
plt.title("Training Split(%) v/s Test Accuracy")
plt.show()


# 2nd part (Variation with Depth) Assuming 70% Training Split
depth = []
acc = []

print("\nDepth \t Accuracy(%)")

for i in range(1,11):
	score = evaluate(dataset, decision_tree, 0.7, i, max_split, 'gini')
	print("{} \t {}%".format(i,score))
	depth.append(i)
	acc.append(score)

plt.plot(depth,acc)
plt.xlabel("Depth")
plt.ylabel("Test Accuracy(%)")
plt.title("Depth v/s Test Accuracy")
plt.show()



###################################################################################

print("Results using Information Gain:\n")
# 1st part (Variation with Train-Test Split)
split = []
acc = []

print("Split(%) \t Accuracy(%)")

for i in range(50,100,5):
	score = evaluate(dataset, decision_tree, i/100, max_depth, max_split, 'info_gain')
	print("{} \t {}%".format(i,score))
	split.append(i)
	acc.append(score)

plt.plot(split,acc)
plt.xlabel("Split(%)")
plt.ylabel("Test Accuracy(%)")
plt.title("Training Split(%) v/s Test Accuracy")
plt.show()


# 2nd part (Variation with Depth) Assuming 70% Training Split
depth = []
acc = []

print("\nDepth \t Accuracy(%)")

for i in range(1,11):
	score = evaluate(dataset, decision_tree, 0.7, i, max_split, 'info_gain')
	print("{} \t {}%".format(i,score))
	depth.append(i)
	acc.append(score)

plt.plot(depth,acc)
plt.xlabel("Depth")
plt.ylabel("Test Accuracy(%)")
plt.title("Depth v/s Test Accuracy")
plt.show()
