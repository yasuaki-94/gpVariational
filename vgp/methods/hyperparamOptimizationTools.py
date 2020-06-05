import numpy as np
import pandas as pd
import math
import random
import os
from functools import reduce
from ..methods import vgpTools

#Compute Inverse of the marginal covariance matrix of Nystrom-based GP
#based on Woodbury formula.
def woodburyVGP(varErr, Knm, Kmn, Kmm, KmmInv):
    N = Knm.shape[0]
    M = Knm.shape[1]

    precisionErr = 1/varErr

    leftMat = precisionErr*np.eye(N)

    invTerm = np.linalg.inv(varErr*Kmm + np.dot(Kmn, Knm))
    rightMat = precisionErr * reduce(np.dot, [Knm, invTerm, Kmn])

    return leftMat + rightMat


#Compute the derivative of a covariance matrix K w.r.t hyperparams
def dK_Theta_sqExp(K, Dsq, sigmaSqf, lscale, varErr):
    dK_sigmaSqf = (1/sigmaSqf) * K
    dK_lscale = (1/(2*lscale**2)) * np.multiply(Dsq, K)

    return dK_sigmaSqf, dK_lscale


#Compute the derivative of Nystrom covariance matrix w.r.t hyperparams
def dNystromMat_Theta_sqExp(Knm, Kmn, KmmInv, dKmm_sigmaSqf, dKmm_lscale, dKmn_sigmaSqf, dKmn_lscale):
    dNystromMat_sigmaSqf = 2*reduce(np.dot, [Knm, KmmInv, dKmn_sigmaSqf]) + \
                            reduce(np.dot, [Knm, KmmInv, dKmm_sigmaSqf, KmmInv, Kmn])

    dNystromMat_lscale = 2*reduce(np.dot, [Knm, KmmInv, dKmn_lscale]) + \
                            reduce(np.dot, [Knm, KmmInv, dKmm_lscale, KmmInv, Kmn])

    return dNystromMat_sigmaSqf, dNystromMat_lscale


#Compute the derivative of approximate marginal log likelihood F w.r.t hyperparams
def compute_dF_Theta_sqExp(y, Cov_marginal_inv, dNystrom_sigmaSqf, dNystrom_lscale, varErr):
    N = y.shape[0]

    t_y = np.transpose(y)

    ty_Cov_marginal_inv = np.dot(t_y, Cov_marginal_inv)
    ty_Cov_marginal_inv_transpose = np.transpose(ty_Cov_marginal_inv)


    trace_term_dF_sigmaSq = 0
    trace_term_dF_lscale = 0
    for i in range(N):
        trace_term_dF_sigmaSq += np.dot(Cov_marginal_inv[i,:], dNystrom_sigmaSqf[:,i])
        trace_term_dF_lscale += np.dot(Cov_marginal_inv[i,:], dNystrom_lscale[:,i])

    dF_sigmaSqf = (1/2) * reduce(np.dot, [ty_Cov_marginal_inv, dNystrom_sigmaSqf, ty_Cov_marginal_inv_transpose]) - \
                        (1/2)*trace_term_dF_sigmaSq - \
                        (1/(2*varErr))*(N + np.trace(dNystrom_sigmaSqf))

    dF_lscale = (1/2) * reduce(np.dot, [ty_Cov_marginal_inv, dNystrom_lscale, ty_Cov_marginal_inv_transpose]) - \
                        (1/2) * trace_term_dF_lscale - \
                        (1/(2*varErr)) * (np.trace(dNystrom_lscale))

    dF_varErr = (1/2) * np.dot(ty_Cov_marginal_inv, ty_Cov_marginal_inv_transpose) - \
                    (1/2) * np.trace(Cov_marginal_inv)

    return dF_sigmaSqf, dF_lscale, dF_varErr


#Carry out the task of computing the derivative of F
def dF_Theta_SqExp(y, Knm, Kmn, Kmm, KmmInv, Dmm, Dmn, sigmaSqf, lscale, varErr):
    N = Knm.shape[0]
    M = Knm.shape[1]

    t_y = np.transpose(y)

    Cov_marginal_inv = woodburyVGP(varErr, Knm, Kmn, Kmm, KmmInv)

    dKmn_sigmaSqf, dKmn_lscale = dK_Theta_sqExp(Kmn, Dmn, sigmaSqf, lscale, varErr)
    dKmm_sigmaSqf, dKmm_lscale = dK_Theta_sqExp(Kmm, Dmm, sigmaSqf, lscale, varErr)

    dNystrom_sigmaSqf, dNystrom_lscale = \
        dNystromMat_Theta_sqExp(Knm, Kmn, KmmInv, dKmm_sigmaSqf, dKmm_lscale, dKmn_sigmaSqf, dKmn_lscale)

    dF_sigmaSqf, dF_lscale, dF_varErr = \
        compute_dF_Theta_sqExp(y, Cov_marginal_inv, dNystrom_sigmaSqf, dNystrom_lscale, varErr)

    g = np.array([dF_sigmaSqf, dF_lscale, dF_varErr])

    return g.reshape((3,1))

#Compute the determinant of approximate marginal covariance matrix
def det_Cov_vgp_marginal(varErr, Knm, Kmm, KmmInv, Kmn):
    N = Knm.shape[0]
    return varErr**N * \
                np.linalg.det(Kmm + (1/varErr)*np.dot(Kmn, Knm)) * \
                np.linalg.det(KmmInv)

#Compute F based on the given hyperparams
def F_vgp(y, Knm, Kmm, KmmInv, Kmn, sigmaSqf, lscale, varErr):
    N = Knm.shape[0]
    M = Knm.shape[1]

    y = y.reshape((N,1))
    t_y = np.transpose(y)

    Cov_Nystrom = reduce(np.dot, [Knm, KmmInv, Kmn])
    Cov_marginal = Cov_Nystrom + varErr * np.eye(N)

    logMarginalLik = -(N/2)*math.log(2*math.pi) - \
        (1/2)*(det_Cov_vgp_marginal(varErr, Knm, Kmm, KmmInv, Kmn)) - \
        (1/2)*reduce(np.dot, [t_y, Cov_marginal, y])


    tracePostCov = (1/(2*varErr))*(N*sigmaSqf + np.trace(Cov_Nystrom))

    F = logMarginalLik + tracePostCov

    return F


#Carry out gradient optimization for hyperparams
def hyperparam_sqExp_grad_ascent(y, Dmm, Dnm, Dmn):

    N = Dnm.shape[0]
    M = Dnm.shape[1]

    y = y.reshape((N,1))

    Theta = np.array([1,1,1]).reshape((3,1))

    k = 0
    alpha = 1

    F = []
    Knm = sqExpKernel(Dnm, Theta[0], Theta[1])
    Kmn = np.transpose(Knm)
    Kmm = sqExpKernel(Dmm, Theta[0], Theta[1])
    KmmInv = np.linalg.inv(Kmm)
    F.append(F_vgp(y, Knm, Kmm, KmmInv, Kmn, Theta[0], Theta[1], Theta[2]))


    while True:
        k = k + 1

        Knm = sqExpKernel(Dnm, Theta[0], Theta[1])
        Kmn = np.transpose(Knm)
        Kmm = sqExpKernel(Dmm, Theta[0], Theta[1])
        KmmInv = np.linalg.inv(Kmm)

        g_k = dF_Theta_SqExp(y, Knm, Kmn, Kmm, KmmInv, Dmm, Dmn, Theta[0], Theta[1], Theta[2])

        Theta = Theta + alpha * g_k

        F_new = F_vgp(y, Knm, Kmm, KmmInv, Kmn, Theta[0], Theta[1], Theta[2])

        F.append(F_new)

        print("F=" + str(F_new))

        if np.linalg.norm(F[k] - F[k-1]) < 0.001 or k >= 10:
            return Theta

        alpha = alpha/1.1


#Take the data and obtain hyperparams based on gradient optimization
def hyperparam_est_grad(X, Xm, y):
    Dmm = distSqMat(Xm, Xm)
    Dnm = distSqMat(X, Xm)
    Dmn = np.transpose(Dnm)

    Theta = hyperparam_sqExp_grad_ascent(y, Dmm, Dnm, Dmn)

    sigmaSqf = Theta[0]
    lscale = Theta[1]
    varErr = Theta[2]

    return sigmaSqf, lscale, varErr


#Take the data and obtain hyperparams based on grid search
def hyperparam_est_grid_search(X, Xm, y):

    sigmaSqf = np.array([0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999])
    lenSigmaSqf = sigmaSqf.shape[0]
    lscale = np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1])
    lenLscale = lscale.shape[0]

    index_maxF = 0
    maxF = 0

    for i in range(lenSigmaSqf):
        for j in range(lenLscale):
            Knm = vgpTools.sqExpKernelCrossCov(X, Xm, sigmaSqf[i], lscale[j])
            Kmn = np.transpose(Knm)
            Kmm = vgpTools.sqExpKernelCov(Xm, sigmaSqf[i], lscale[j])
            KmmInv = np.linalg.inv(Kmm)

            F = F_vgp(y, Knm, Kmm, KmmInv, Kmn, sigmaSqf[i], lscale[j], 1)[0,0]
            print("F(X|sigmaSqf={}, lscale={}, varErr={})={}".format(sigmaSqf[i], lscale[j], 1, F))

            if (F > maxF or (i == 0 and j==0)) and not math.isnan(F):
                 index_maxF_sigmaSqf = i
                 index_maxF_lscale = j
                 maxF = F

    print("sigmaSqf={}, lscale={}, varErr={}".format(sigmaSqf[index_maxF], lscale[index_maxF], 1))

    return sigmaSqf[index_maxF], lscale[index_maxF], 1
