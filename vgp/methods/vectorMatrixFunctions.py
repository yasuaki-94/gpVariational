import numpy as np
import pandas as pd
import math
import random
import os

#Normalize the columns of the given (data) matrix X
def normalizeCols(X, mu=None, sdVec=None):
    if mu is None:
        mu = np.mean(X, axis=0)

    if sdVec is None:
        sdVec = np.std(X, axis=0)

    Xnorm = np.zeros(X.shape)
    if len(X.shape)==2:
        for j in range(X.shape[1]):
            Xnorm[:,j] = (X[:,j]-float(mu[j]))/float(sdVec[j])
        return Xnorm
    elif len(X.shape)==1:
        return centerVals(X, mu=float(mu))/float(sdVec)

#Center the value of the given vector x around its mean
def centerVals(x, mu=None):
    if mu is None:
        mu = float(np.mean(x))
    return x - mu

#Compute the matrix of square of the distance between the points of X1 and X2
def distSqMat(X1, X2):
    N1 = X1.shape[0]
    N2 = X2.shape[0]
    d = X2.shape[1]

    Dsq = np.zeros((N1,N2))

    for i in range(N1):
        x1_i = X1[i,:].reshape((1,d))
        diffMat_i = x1_i - X2
        Dsq[i,:] = np.diagonal(np.dot(diffMat_i, np.transpose(diffMat_i)))

    return Dsq
