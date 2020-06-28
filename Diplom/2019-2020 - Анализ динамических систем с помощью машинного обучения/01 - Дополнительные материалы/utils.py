from random import sample
from numpy.random import uniform
from math import isnan
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

def hopkins(X):
    d = X.shape[1]
    #d = len(vars) # columns
    n = len(X) # rows
    m = int(0.1 * n) # heuristic from article [1]
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
 
    rand_X = sample(range(0, n, 1), m)
 
    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])
 
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print (ujd, wjd)
        H = 0
    return H

def dbscan_model(df, eps, samples):
    dbscan = DBSCAN(eps=eps, min_samples=samples)
    dbscan.fit(df)
    return dbscan.labels_

def EM_model(df, n, dis_type):
    gmm = GaussianMixture(n_components=30, covariance_type=dis_type).fit(df)
    return gmm.predict(df)

def Aglo_model(df, n, linkage):
    clustering = AgglomerativeClustering(n, linkage=linkage).fit(df)
    return clustering.labels_
    