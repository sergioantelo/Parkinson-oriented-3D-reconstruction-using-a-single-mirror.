import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# PCA component reduction and function fitting
x1,y1,z1 = pickle.load(open("static","rb"))
#x2,y2,z2 = pickle.load(open("coords_y","rb"))
#x3,y3,z3 = pickle.load(open("coords_z","rb"))
#x4,y4,z4 = pickle.load(open("coords_pend","rb"))

#data = pd.read_table('results/results.csv', sep=';', decimal=b',')
#data = data.as_matrix()

# for each of the lines. != distances
X = np.array([x1,y1,z1]).T
p0 = X[0]
p1 = X[-1]
d = np.sqrt(np.sum((p0-p1)**2))
print(d)

X = np.array([x4,y4,z4]).T
pca = PCA(n_components=2)
X = pca.fit_transform(X)

poly = PolynomialFeatures(degree=2)
X_ = poly.fit_transform(X[:, 0:1])

clf = linear_model.LinearRegression()
clf.fit(X_, X[:, 1:2])
fitted_y = clf.predict(X_)

plt.plot(X[:, 0], X[:, 1])
plt.plot(X[:, 0], fitted_y)

RMS = np.sqrt(np.mean((X[:, 1] - fitted_y) ** 2))
print(RMS)
