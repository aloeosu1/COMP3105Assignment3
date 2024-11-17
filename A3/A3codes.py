# Michael Han 101157504
# Azamat (Rush) Galimov 101263850
# COMP3105A
# Assignment 3

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import logsumexp
import cvxopt
import pandas as pd
from scipy.spatial.distance import cdist
from A3helpers import generateData, augmentX


# 1a
def minMulDev(X, Y):
    n, d = X.shape
    _, k = Y.shape
    def objective(W_flat):

        # repshaping weights to d x k
        W = W_flat.reshape(d, k)

        # predicted values of Y
        Y_pred = X @ W

        # calculating multinomial deviance loss
        loss = np.sum(logsumexp(Y_pred, axis=1) - np.sum(Y_pred * Y, axis=1)) / n
        return loss

    # intital weights
    initial_W = np.zeros(d * k)

    # optimizing objective function
    result = minimize(fun=objective, x0=initial_W, method='BFGS')

    # reshape optimized weights to d x k and return
    return result.x.reshape(d, k)
    

# 1b
def classify(Xtest, W):
    #get class scores for each sample
    scores = Xtest @ W

    # return the class with highest predicted score
    return np.eye(W.shape[1])[np.argmax(scores, axis=1)]
    

# 1c
def calculateAcc(Yhat, Y):
    # getting number of correct predictions
    correct_predictions = np.sum(np.all(Yhat == Y, axis=1))
    total_predictions = Y.shape[0]

    # calculate and return acc
    acc = correct_predictions / total_predictions
    return acc



# 2a
def PCA(X, k):
    # compute mean
    mean = np.mean(X, axis=0)

    # make X centered by subtracting mean from X
    X = X - mean

    # getting covariance matrix 
    covariance_matrix = np.dot(X.T, X)

    # getting eigenvectors with top k largest eigenvalues (ascending order, so get subset of the last k elements (largest))
    eigenvalues, eigenvectors = scipy.linalg.eigh(X.T @ X, subset_by_index=[covariance_matrix.shape[0] - k, covariance_matrix.shape[0] - 1])

    # transpose to make rows correspond to top k projecting directions
    U = eigenvectors.T

    return U

# 2c
def projPCA(Xtest, mu, U):
    # proj = (Xtest - mu.T)U.T
    return np.dot((Xtest - mu.T), U.T)


# 2d
def synClsExperimentsPCA():
    n_runs = 100
    n_train = 128
    n_test = 1000
    dim_list = [1, 2]
    gen_model_list = [1, 2]
    train_acc = np.zeros([len(dim_list), len(gen_model_list), n_runs])
    test_acc = np.zeros([len(dim_list), len(gen_model_list), n_runs])
    for r in range(n_runs):
        for i, k in enumerate(dim_list):
            for j, gen_model in enumerate(gen_model_list):
                Xtrain, Ytrain = generateData(n=n_train, gen_model=gen_model)
                Xtest, Ytest = generateData(n=n_test, gen_model=gen_model)
                U = PCA(Xtrain, k)

                # calulate mean
                mu = np.mean(Xtrain, axis=0)

                # call your projPCA to find the new features
                Xtrain_proj = projPCA(Xtrain, mu, U)
                # call your projPCA to find the new features
                Xtest_proj = projPCA(Xtest, mu, U)

                Xtrain_proj = augmentX(Xtrain_proj) # add augmentation
                Xtest_proj = augmentX(Xtest_proj)

                W = minMulDev(Xtrain_proj, Ytrain) # from Q1
                Yhat = classify(Xtrain_proj, W) # from Q1
                train_acc[i, j, r] = calculateAcc(Yhat, Ytrain) # from Q1
                Yhat = classify(Xtest_proj, W)
                test_acc[i, j, r] = calculateAcc(Yhat, Ytest)

    # compute the average accuracies over runs
    avg_train_acc = np.mean(train_acc, axis=2)
    avg_test_acc = np.mean(test_acc, axis=2)

    # return 2-by-2 train accuracy and 2-by-2 test accuracy  
    return avg_train_acc, avg_test_acc



# 3a
def kmeans(X, k, max_iter=1000):
    n, d = X.shape
    assert max_iter > 0

    # Choose k random points from X as initial centers
    # randomly select k points for initial cluster centers
    random = np.random.default_rng()
    U = X[random.choice(n, k, replace = False), :]

    for i in range(max_iter):

        # Compute pairwise distance between X and U
        D = cdist(X, U, metric = 'sqeuclidean')

        # Find the new cluster assignments
        Y = np.zeros((n, k))
        #assign each point to the closest cluster
        Y[np.arange(n), np.argmin(D, axis = 1)] = 1

        old_U = U

        # Update cluster centers 
        # weighted sum of data points in cluster
        U = np.linalg.solve(Y.T @ Y + 1e-8 * np.eye(k), Y.T @ X)

        if np.allclose(old_U, U):
            break

        # Compute objective value
        obj_val = (0.5 / n) * np.sum((X - Y @ U) ** 2) 

        return Y, U, obj_val
    

# 3b
def repeatKmeans(X, k, n_runs=100):
    best_obj_val = float('inf')
    for r in range(n_runs):
        Y, U, obj_val = kmeans(X, k)
        # Compare obj_val with best_obj_val. If it is lower,
        if obj_val < best_obj_val:
            # then record the current Y, U and update best_obj_val
            best_obj_val = obj_val
            best_Y, best_U = Y, U

        
    # Return the best Y, U and best_obj_val
    return best_Y, best_U, best_obj_val

# 3c
def chooseK(X, k_candidates=[2,3,4,5,6,7,8,9]):
    obj_val_list = []

    # call repeatKmeans for each k
    for k in k_candidates:
        Y, U, obj_val = repeatKmeans(X, k)
        # add current obj_val to obj_val_list
        obj_val_list.append(obj_val)

    return obj_val_list

