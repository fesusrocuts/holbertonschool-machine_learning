#!/usr/bin/env python3
""" 10. Hello, sklearn!"""


import sklearn.cluster


def kmeans(X, k):
    """ 10. Hello, sklearn!"""
    kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    return kmeans.cluster_centers_, kmeans.labels_
