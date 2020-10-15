import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

def L2norm(X1, X2):		#Function to calculate L2 Norm
	distance = 0
	for i in range(len(X1)):
		distance += (X1[i] - X2[i])**2

	distance = distance**0.5
	return distance

def distance_matrix(X):
	matrix = np.ndarray(shape=(len(X),len(X)))
	# print(matrix)
	for i in range(len(X)):
		for j in range(i,len(X)):
			if i == j:
				matrix[i][j] = sys.maxsize
			else:
				norm = L2norm(X[i], X[j])
				matrix[i][j] = norm
				matrix[j][i] = norm

	return matrix





def find_clusters(input,linkage):
    clusters = {}
    row_index = -1
    col_index = -1
    array = []
    

    for n in range(input.shape[0]):
        array.append(n)
        
    clusters[0] = array.copy()

    #finding minimum value from the distance matrix
    #note that this loop will always return minimum value from bottom triangle of matrix
    for k in range(1, input.shape[0]):
        min_val = sys.maxsize
        
        for i in range(0, input.shape[0]):
            for j in range(0, input.shape[1]):
                if(input[i][j]<=min_val):
                    min_val = input[i][j]
                    row_index = i
                    col_index = j
                    
        #once we find the minimum value, we need to update the distance matrix
        #updating the matrix by calculating the new distances from the cluster to all points
        
        #for Single Linkage
        if(linkage == "single" or linkage =="Single"):
            for i in range(0,input.shape[0]):
                if(i != col_index):
                    #we calculate the distance of every data point from newly formed cluster and update the matrix.
                    temp = min(input[col_index][i],input[row_index][i])
                    #we update the matrix symmetrically as our distance matrix should always be symmetric
                    input[col_index][i] = temp
                    input[i][col_index] = temp
        #for Complete Linkage
        elif(linkage=="Complete" or linkage == "complete"):
             for i in range(0,input.shape[0]):
                if(i != col_index and i!=row_index):
                    temp = min(input[col_index][i],input[row_index][i])
                    input[col_index][i] = temp
                    input[i][col_index] = temp
        #for Average Linkage
        elif(linkage=="Average" or linkage == "average"):
             for i in range(0,input.shape[0]):
                if(i != col_index and i!=row_index):
                    temp = (input[col_index][i]+input[row_index][i])/2
                    input[col_index][i] = temp
                    input[i][col_index] = temp
                   
        #set the rows and columns for the cluster with higher index i.e. the row index to infinity
        #Set input[row_index][for_all_i] = infinity
        #set input[for_all_i][row_index] = infinity
        for i in range (0,input.shape[0]):
            input[row_index][i] = sys.maxsize
            input[i][row_index] = sys.maxsize
            
        #Manipulating the dictionary to keep track of cluster formation in each step
        #if k=0,then all datapoints are clusters
       
        minimum = min(row_index,col_index)
        maximum = max(row_index,col_index)
        for n in range(len(array)):
            if(array[n]==maximum):
                array[n] = minimum
        clusters[k] = array.copy()
        
    return clusters


def hierarchical_clustering(data,linkage,no_of_clusters):  
    #first step is to calculate the initial distance matrix
    #it consists distances from all the point to all the point
    initial_distances = distance_matrix(data)
    #making all the diagonal elements infinity 
    clusters = find_clusters(initial_distances,linkage) 

    iteration_number = initial_distances.shape[0] - no_of_clusters
    labels = clusters[iteration_number]

    return labels


#####################################################################

# Load the data in Dataframe
df = pd.read_csv("cancer.csv")

# Chose relevant columns
X = df.iloc[:,2:32]

# Convert Dataframe to Array
X = X.values

# Linkage = Single

labels = hierarchical_clustering(X, "single", 4)

x0 = []
y0 = []

x1 = []
y1 = []

x2 = []
y2 = []

x3 = []
y3 = []

unique_labels = list(set(labels))

ind_x = list(df.iloc[:,2:32].columns).index("radius_mean")
ind_y = list(df.iloc[:,2:32].columns).index("texture_mean")

n_cluster1 = 0
n_cluster2 = 0
n_cluster3 = 0
n_cluster4 = 0

for i in range(len(labels)):
	idx = unique_labels.index(labels[i])
	if idx == 0:
		x0.append(X[i][ind_x])
		y0.append(X[i][ind_y])
		n_cluster1 += 1
	elif idx == 1:
		x1.append(X[i][ind_x])
		y1.append(X[i][ind_y])
		n_cluster2 += 1
	elif idx == 2:
		x2.append(X[i][ind_x])
		y2.append(X[i][ind_y])
		n_cluster3 += 1
	elif idx == 3:
		x3.append(X[i][ind_x])
		y3.append(X[i][ind_y])
		n_cluster4 += 1

plt.scatter(x0, y0, marker='o', color='red', label="Label 0, n="+str(n_cluster1))
plt.scatter(x1, y1, marker='o', color='blue', label="Label 1, n="+str(n_cluster2))
plt.scatter(x2, y2, marker='o', color='green', label="Label 0, n="+str(n_cluster3))
plt.scatter(x3, y3, marker='o', color='black', label="Label 1, n="+str(n_cluster4))

plt.xlabel("radius_mean")
plt.ylabel("texture_mean")

plt.legend()

plt.title("Hierarchial Clusters (n=4) (Linkage=Single)")
plt.show()

# Linkage = Complete


labels = hierarchical_clustering(X, "complete", 4)

x0 = []
y0 = []

x1 = []
y1 = []

x2 = []
y2 = []

x3 = []
y3 = []

unique_labels = list(set(labels))

ind_x = list(df.iloc[:,2:32].columns).index("radius_mean")
ind_y = list(df.iloc[:,2:32].columns).index("texture_mean")

n_cluster1 = 0
n_cluster2 = 0
n_cluster3 = 0
n_cluster4 = 0

for i in range(len(labels)):
	idx = unique_labels.index(labels[i])
	if idx == 0:
		x0.append(X[i][ind_x])
		y0.append(X[i][ind_y])
		n_cluster1 += 1
	elif idx == 1:
		x1.append(X[i][ind_x])
		y1.append(X[i][ind_y])
		n_cluster2 += 1
	elif idx == 2:
		x2.append(X[i][ind_x])
		y2.append(X[i][ind_y])
		n_cluster3 += 1
	elif idx == 3:
		x3.append(X[i][ind_x])
		y3.append(X[i][ind_y])
		n_cluster4 += 1

plt.scatter(x0, y0, marker='o', color='red', label="Label 0, n="+str(n_cluster1))
plt.scatter(x1, y1, marker='o', color='blue', label="Label 1, n="+str(n_cluster2))
plt.scatter(x2, y2, marker='o', color='green', label="Label 0, n="+str(n_cluster3))
plt.scatter(x3, y3, marker='o', color='black', label="Label 1, n="+str(n_cluster4))

plt.xlabel("radius_mean")
plt.ylabel("texture_mean")

plt.legend()

plt.title("Hierarchial Clusters (n=4) (Linkage=Complete)")
plt.show()

# Linkage = Average

labels = hierarchical_clustering(X, "average", 4)

x0 = []
y0 = []

x1 = []
y1 = []

x2 = []
y2 = []

x3 = []
y3 = []

unique_labels = list(set(labels))

ind_x = list(df.iloc[:,2:32].columns).index("radius_mean")
ind_y = list(df.iloc[:,2:32].columns).index("texture_mean")

n_cluster1 = 0
n_cluster2 = 0
n_cluster3 = 0
n_cluster4 = 0

for i in range(len(labels)):
	idx = unique_labels.index(labels[i])
	if idx == 0:
		x0.append(X[i][ind_x])
		y0.append(X[i][ind_y])
		n_cluster1 += 1
	elif idx == 1:
		x1.append(X[i][ind_x])
		y1.append(X[i][ind_y])
		n_cluster2 += 1
	elif idx == 2:
		x2.append(X[i][ind_x])
		y2.append(X[i][ind_y])
		n_cluster3 += 1
	elif idx == 3:
		x3.append(X[i][ind_x])
		y3.append(X[i][ind_y])
		n_cluster4 += 1

plt.scatter(x0, y0, marker='o', color='red', label="Label 0, n="+str(n_cluster1))
plt.scatter(x1, y1, marker='o', color='blue', label="Label 1, n="+str(n_cluster2))
plt.scatter(x2, y2, marker='o', color='green', label="Label 0, n="+str(n_cluster3))
plt.scatter(x3, y3, marker='o', color='black', label="Label 1, n="+str(n_cluster4))

plt.xlabel("radius_mean")
plt.ylabel("texture_mean")

plt.legend()

plt.title("Hierarchial Clusters (n=4) (Linkage=Average)")

plt.show()