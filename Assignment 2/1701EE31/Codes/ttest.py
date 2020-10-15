import numpy as np
import pandas as pd


def get_parameters(X):
	# mean = X.mean()
	# var = X.var()
	# n = len(X)
	# print(type(X))
	mean = np.mean(pd.to_numeric(X[X.columns[0]]))
	var = np.var(pd.to_numeric(X[X.columns[0]]))
	n = len(X)

	# print(mean)

	return mean, var, n

def get_t_stats(mu1, var1, n1, mu2, var2, n2):
	t_stat = abs(mu1 - mu2)/np.sqrt((var1/n1)+(var2/n2))

	return t_stat

def get_t_test_result(X1, X2):
	mu1, var1, n1 = get_parameters(X1)
	mu2, var2, n2 = get_parameters(X2)

	t_score = get_t_stats(mu1, var1, n1, mu2, var2, n2)

	return t_score

def get_features(X, y):
	class_name = pd.DataFrame(y).columns[0]

	unique_labels = pd.DataFrame(y)[class_name].unique()

	features = X.columns
	# print(features)
	df = X
	df["class"] = pd.DataFrame(y)
	# print(y)
	f = []
	fi = []

	for feature in features:
		tmp = []

		for label in unique_labels:
			X1 = df.loc[df["class"] == label][feature]
			X2 = df.loc[df["class"] != label][feature]

			# print(X1,X2)
			# print("\n")
			# print(pd.DataFrame(X1).mean())
			# print("\n")
			X1 = pd.DataFrame(X1)
			X2 = pd.DataFrame(X2)

			t_score = get_t_test_result(X1, X2)
			tmp.append(t_score)
			# print(X1.mean())
			# f.append(feature)
			# fi.append(t_)
		# print(tmp)
		fi.append(max(tmp))
		f.append(feature)

	return f, fi
