import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
plt.style.use('seaborn-whitegrid')

def L2norm(X1, X2):		#Function to calculate L2 Norm
	distance = 0
	for i in range(len(X1)):
		distance += (X1[i] - X2[i])**2

	distance = distance**0.5
	return distance


def update_labels(X, point, eps, Minpts, labels, cluster_val):
	neighbors = []
	label_index = []
	_labels = labels

	for i in range(0, X.shape[0]):
		if np.linalg.norm(X[point] - X[i]) < eps:
			neighbors.append(X[i])
			label_index.append(i)

	if len(neighbors) < Minpts:
		for j in range(len(_labels)):
			if j in label_index:
				_labels[j] = -1
	else:
		for j in range(len(_labels)):
			if j in label_index:
				_labels[j] = cluster_val

	return _labels


#############################################################


# Load the data in Dataframe
df = pd.read_csv("../cancer.csv")

# Chose relevant columns
X = df.iloc[:,2:32]

# Convert Dataframe to Array
X = X.values

labels = [0]*len(X)
C = 1 

eps = 100
Minpts = 3

eps = float(input("Enter the value of Eps\n"))
Minpts = float(input("Enter the value of Minpts\n"))

for p in range(0, X.shape[0]):
	if labels[p] == 0:
		labels = update_labels(X, p, eps, Minpts, labels, C)
		C = C+1

# print(labels)


scatterColors = ['green', 'brown', 'red', 'purple', 'orange', 'yellow']

unique_labels = list(set(labels))

# unique_labels = [-1].extend(range(len(unique_labels)-1))


ind_x = list(df.iloc[:,2:32].columns).index("radius_mean")
ind_y = list(df.iloc[:,2:32].columns).index("texture_mean")


for i in range(len(unique_labels)):
	if unique_labels[i] == -1:
		color = 'black'
	else:
		color = scatterColors[i]

	x = []
	y = [] 

	for j in range(len(X)):
		if labels[j] == unique_labels[i]:
			x.append(X[j][ind_x])
			y.append(X[j][ind_y])
	plt.scatter(x, y, c=color, alpha=1, marker='.', label="Label {}, n={}".format(unique_labels[i], len(x)))


plt.xlabel("radius_mean")
plt.ylabel("texture_mean")

plt.legend()

plt.title("DBSCAN Cluster (eps={}) (Minpts={})".format(eps, Minpts))

plt.show()


# dbs = DBSCAN(eps=eps, min_samples=Minpts, metric='l2').fit(X)

# print(dbs.labels_)