from random import seed
from random import randrange
from csv import reader
import numpy as np
# from sklearn.model_selection import train_test_split
import random
import math


# Load a CSV file
def load_csv(filename):
	file = open(filename, "rt")
	lines = reader(file)
	dataset = list(lines)
	return dataset
 
# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())



# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right
 
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Calculate the Information Gain for a split dataset
def information_gain(groups, classes):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	ig = 0.0
	entropy_init = 0.0

	whole_data = []
	for group in groups:
		whole_data.extend(group)
	# print(whole_data)
	for class_val in classes:
		p = [row[-1] for row in whole_data].count(class_val) / n_instances
		entropy_init += (-1 * p * math.log2(p))

	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			# print(p)
			if p == 0:
				continue
			score += (-1 * p * math.log2(p))*(len(group)/n_instances)
		# weight the group score by its relative size
		ig += entropy_init - score
	return ig



# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	gini = 0.0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_instances)
	return gini
 
# Select the best split point for a dataset
def get_split(dataset, split):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset[0])-1):
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			if split == 'gini':
				gini = gini_index(groups, class_values)
			else:
				gini = information_gain(groups, class_values)

			# gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}
 
# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)
 
# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth, criterion):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left, criterion)
		split(node['left'], max_depth, min_size, depth+1, criterion)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right,criterion)
		split(node['right'], max_depth, min_size, depth+1, criterion)
 
# Build a decision tree
def build_tree(train, max_depth, min_size, criterion):
	root = get_split(train, criterion)
	split(root, max_depth, min_size, 1, criterion)
	return root

# Make a prediction with a decision tree
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

# Classification and Regression Tree Algorithm
def decision_tree(train, test, max_depth, min_size, criterion):
	tree = build_tree(train, max_depth, min_size, criterion)
	predictions = list()
	for row in test:
		prediction = predict(tree, row)
		predictions.append(prediction)
	return(predictions)

# Split a dataset into a train and test set
def train_test_split(dataset, data_split):
	train = list()
	train_size = data_split * len(dataset)
	dataset_copy = list(dataset)
	while len(train) < train_size:
		index = randrange(len(dataset_copy))
		train.append(dataset_copy.pop(index))
	return train, dataset_copy


# Evaluate an algorithm
def evaluate(dataset, algorithm, data_split, max_depth, min_size, criterion='gini'):
	train, test = train_test_split(dataset, data_split=data_split)
	predicted = algorithm(train, test, max_depth, min_size, criterion)
	actual = [row[-1] for row in test]
	# print(len(actual),len(predicted))
	accuracy = accuracy_metric(actual, predicted)

	return accuracy

# filename = 'heart.csv'
# dataset = load_csv(filename)[1:]

# # convert string attributes to integers
# for i in range(len(dataset[0])):
# 	str_column_to_float(dataset, i)

# # print(dataset)
# score = evaluate(dataset, decision_tree, 0.3, 3, 5, 'info_gain')
# print(score)