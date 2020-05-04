import numpy as np
import pandas as pd
import math
import random
import os
from functools import reduce
from datetime import datetime

def sqExpKernel(diff, sigmaSqf, lscale):
    sqNorm = (np.linalg.norm(diff))**2
    expTerm = math.exp(-sqNorm/(2*lscale))
    return sigmaSqf * expTerm

def sqExpKernelCov(X, sigmaSqf, lscale):
    N = X.shape[0]
    d = X.shape[1]

    K = np.zeros((N, N))

    def kernFunc(diff):
        return sqExpKernel(diff, sigmaSqf, lscale)

    for i in range(N):
        x_i = X[i,:].reshape((1,d))
        diffMat_i_iToN = x_i - X[i:N,:]
        K[i,i:] = np.apply_along_axis(kernFunc, 1, diffMat_i_iToN)

    K = K + np.transpose(K)

    return K


def sqExpKernelCrossCov(X1, X2, sigmaSqf, lscale):
    N = X1.shape[0]
    M = X2.shape[0]
    d = X2.shape[1]

    K = np.zeros((N, M))

    def kernelFunc(diff):
        return sqExpKernel(diff, sigmaSqf, lscale)

    for i in range(N):
        x1_i = X1[i,:].reshape((1,d))
        diffMat_i = x1_i - X2
        kernelVec_i = np.diagonal(np.dot(diffMat_i, np.transpose(diffMat_i)))
        kernelVec_i = np.exp(-kernelVec_i/(2*lscale))
        K[i,:] = kernelVec_i

    return K


def paramsEst(X, y, Xm, sigmaSqf, lscale, varErr, printElapsedTime):
    start = datetime.now()

    N = X.shape[0]
    d = X.shape[1]
    M = Xm.shape[0]
    precErr = varErr**(-1)

    y = y.reshape((N,1))
    Kmm = sqExpKernelCov(Xm, sigmaSqf, lscale)
    Knm = sqExpKernelCrossCov(X, Xm, sigmaSqf, lscale)
    Kmn = np.transpose(Knm)


    SigmaInv = Kmm + precErr * np.dot(Kmn, Knm)
    Sigma = np.linalg.inv(SigmaInv)

    mu = precErr * reduce(np.dot, [Kmm, Sigma, Kmn, y])
    A = reduce(np.dot, [Kmm, Sigma, Kmm])

    timeDiff = datetime.now() - start

    if printElapsedTime:
        print("Elapsed Time (in Seconds): " + str(timeDiff.seconds))

    return mu, A

def predictMean(X, Xm, mu, y, sigmaSqf, lscale, printErr, \
                scalingParams={"mean": 0, "sd":1}):
    M = Xm.shape[0]
    N = X.shape[0]
    y = y.reshape((N, 1))

    Kxm = sqExpKernelCrossCov(X, Xm, sigmaSqf, lscale)
    KmmInv = np.linalg.inv(sqExpKernelCov(Xm, sigmaSqf, lscale))
    mu = mu.reshape((M,1))

    predictedMean = reduce(np.dot, [Kxm, KmmInv, mu]).reshape((N, 1))
    predictedMean = predictedMean*scalingParams["sd"] + scalingParams["mean"]


    if printErr == True:
        rmse  = np.linalg.norm(y - predictedMean)/math.sqrt(N)
        print("Test RMSE={}.".format(rmse))

    return predictedMean
