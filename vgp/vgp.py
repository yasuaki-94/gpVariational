import numpy as np
import pandas as pd
import math
import random
import os
from functools import reduce
from datetime import datetime
import sys
from .methods import vgpTools
from .methods import vectorMatrixFunctions
from .methods import hyperparamOptimizationTools


class VGPsqExp:
    """
    Create object for variational GP with squared exponential
    kernel covariance function.

    Arguments:
    X -- Matrix for predictor values in the training data
    y -- Vector of response values
    Xm -- Matrix for inducing points
    sigmaSqf, lscale, varErr -- Hyperparameters for the model
    normalizeData -- Whether or not to normalize data for parameter estimation
    """

    def __init__(self, X, y, Xm, sigmaSqf=None, lscale=None, varErr=None, normalizeData=True):
        self.normalizeData = normalizeData
        self.dataInfo = { \
            "Xmean": np.mean(X, axis=0), \
            "Xsd": np.std(X, axis=0), \
            "ymean": np.mean(y, axis=0), \
            "ysd": np.std(y, axis=0) \
        }

        if self.normalizeData:
            self.X = vectorMatrixFunctions.normalizeCols(X, \
                                  self.dataInfo["Xmean"], \
                                  self.dataInfo["Xsd"])

            self.Xm = vectorMatrixFunctions.normalizeCols(Xm, \
                                  self.dataInfo["Xmean"], \
                                  self.dataInfo["Xsd"])

            self.y = vectorMatrixFunctions.normalizeCols(y, \
                                  self.dataInfo["ymean"], \
                                  self.dataInfo["ysd"])
        else:
            self.X = X
            self.Xm = Xm
            self.y = y


        self.hyperparams = {\
            "sigmaSqf": sigmaSqf,\
            "lscale": lscale,\
            "varErr": varErr\
        }

        self.variationalParams = {\
            "mu": np.zeros((self.Xm.shape[0], 1)), \
            "A": np.identity(self.Xm.shape[0]) \
        }

    def optimizeHyperparams(self, method="grid"):
        """
        Optimize the hyperparameters

        Arguments:
        method: Method to be used for hyperparameter optimization.
                Default is set to be grid search (grid).
        """

        if method == "grid":
            self.hyperparams["sigmaSqf"], \
            self.hyperparams["lscale"], \
            self.hyperparams["varErr"] = \
                hyperparamOptimizationTools.hyperparam_est_grid_search(self.X, self.Xm, self.y)
        elif method=="grad-ascent":
            self.hyperparams["sigmaSqf"], \
            self.hyperparams["lscale"], \
            self.hyperparams["varErr"] = \
                hyperparamOptimizationTools.hyperparam_est(self.X, self.Xm, self.y)
        else:
            print("There is no method named {}".format(method))

    def train(self, returnParam=False, printElapsedTime=False, returnElapsedTime=False):
        """
        Train the model and estimate parameters in the variational distribution

        Arguments:
        returnParam -- Specify whether or not to return the estimated parameters
        printElapsedTime -- Specify whether or not to show the time taken for training
        """
        self.mu, self.A, elapsedTime = vgpTools.paramsEst(self.X, \
                                    self.y, \
                                    self.Xm, \
                                    self.hyperparams["sigmaSqf"], \
                                    self.hyperparams["lscale"], \
                                    self.hyperparams["varErr"],
                                    printElapsedTime)

        if returnParam and returnElapsedTime:
            return self.mu, self.A, elapsedTime
        elif returnParam:
            return self.mu, self.A
        elif returnElapsedTime:
            return elapsedTime

    def predictMean(self, Xtest, ytest, printErr=False):
        """
        Predict the mean values at given test points

        Arguments:
        Xtest -- Test set points
        ytest -- Response values at test points (used for computing RMSE)
        printErr -- Specify whether or not to show test RMSE.
        """

        if self.normalizeData:
            Xtest = vectorMatrixFunctions.normalizeCols(Xtest,\
                                            self.dataInfo["Xmean"], \
                                            self.dataInfo["Xsd"])

            return vgpTools.predictMean(Xtest,\
                               self.Xm, \
                               self.mu, \
                               ytest,\
                               self.hyperparams["sigmaSqf"], \
                               self.hyperparams["lscale"], \
                               printErr,
                               {"mean": self.dataInfo["ymean"], \
                               "sd": self.dataInfo["ysd"]})
        else:
            return vgpTools.predictMean(Xtest,\
                               self.Xm, \
                               self.mu, \
                               ytest,\
                               self.hyperparams["sigmaSqf"], \
                               self.hyperparams["lscale"], \
                               printErr)
