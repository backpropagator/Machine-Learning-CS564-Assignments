import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


def L2norm(X1, X2):		#Function to calculate L2 Norm
	distance = 0
	for i in range(len(X1)):
		distance += (X1[i] - X2[i])**2

	distance = distance**0.5
	return distance

def calculate_cost(medoid, X):
	cost = 0.0
	for i in range(len(X)):
		cost += L2norm(medoid, X[i])

	return cost

#####################################################################

# Load the data in Dataframe
df = pd.read_csv("cancer.csv")

# Chose relevant columns
X = df.iloc[:,2:32]

# Convert Dataframe to Array
X = X.values


idx1 = np.random.randint(low=0, high=len(X), dtype=int)
idx2 = np.random.randint(low=0, high=len(X), dtype=int)

while idx2 == idx1:
	idx2 = np.random.randint(low=0, high=len(X), dtype=int)

idx1_prev = idx1+1
idx2_prev = idx2+1


# Dictionary to store the cluster
label_dict = {}
label_dict[0] = []
label_dict[1] = []

while idx1 != idx1_prev and idx2 != idx2_prev:
	for i in range(len(X)):
		if i != idx1 and i != idx2:
			d1 = L2norm(X[i], X[idx1])
			d2 = L2norm(X[i], X[idx2])

			if d1 < d2:
				label_dict[0].append(X[i])
			else:
				label_dict[1].append(X[i])


	cost1 = calculate_cost(X[idx1], label_dict[0]) + calculate_cost(X[idx2], label_dict[1])
	
	idx_new = np.random.randint(low=0, high=len(X), dtype=int)

	while idx_new == idx1:
		idx_new = np.random.randint(low=0, high=len(X), dtype=int)

	label_dict[0] = []
	label_dict[1] = []

	for i in range(len(X)):
		if i != idx1 and i != idx2:
			d1 = L2norm(X[i], X[idx_new])
			d2 = L2norm(X[i], X[idx2])

			if d1 < d2:
				label_dict[0].append(X[i])
			else:
				label_dict[1].append(X[i])


	cost2 = calculate_cost(X[idx_new], label_dict[0]) + calculate_cost(X[idx2], label_dict[1])

	if cost2 <= cost1:
		idx1_prev = idx1
		idx2_prev = idx2

		idx1 = idx2
		idx2 = idx_new

		label_dict[0] = []
		label_dict[1] = []
	else:
		break

n_cluster1 = len(label_dict[0])
n_cluster2 = len(label_dict[1])

print("Number of points in Cluster 1: {}".format(n_cluster1))
print("Number of points in Cluster 2: {}".format(n_cluster2))


# Plotting Section

ind_x = list(df.iloc[:,2:32].columns).index("radius_mean")
ind_y = list(df.iloc[:,2:32].columns).index("texture_mean")


x0 = []
y0 = []

x1 = []
y1 = []

for i in range(len(label_dict[0])):
	x0.append(label_dict[0][i][ind_x])
	y0.append(label_dict[0][i][ind_y])

for i in range(len(label_dict[1])):
	x1.append(label_dict[1][i][ind_x])
	y1.append(label_dict[1][i][ind_y])

plt.scatter(x0, y0, marker='o', color='red', label="Label 0, n="+str(n_cluster1))
plt.scatter(x1, y1, marker='o', color='blue', label="Label 1, n="+str(n_cluster2))

plt.xlabel("radius_mean")
plt.ylabel("texture_mean")

plt.legend()

plt.title("k-Medoid Clusters for Cancer Dataset")

plt.show()
