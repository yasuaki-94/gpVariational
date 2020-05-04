import numpy as np
import pandas as pd
import math
import random
import os

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



def centerVals(x, mu=None):
    if mu is None:
        mu = float(np.mean(x))
    return x - mu
