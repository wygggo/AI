#!/usr/bin/env python
# coding=utf-8


from __future__ import print_function

import numpy as np
import matplotlib.pylab as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname = r"/home/AI/mlpractise/font/simsun.ttc", size = 18)

def linearRegression(alpha = 0.01, num_iters = 500):
    print(u"加载数据....\n")
    
    data = loadtxtAndcsv_data("data.csv", ",", np.float64)
    X = data[:,0:-1]
    y = data[:, -1]
    m = len(y)
    col = data.shape[1]

    X, mu, sigma = featureNormaliza(X)
    plot_X1_X2(X)

    X = np.hstack((np.ones((m, 1)), X))
    print(u"\n执行梯度下降算法....\n")

    theta = np.zeros((col, 1))
    y = y.reshape(-1, 1)
    theta, J_history = gradientDescent(X, y, theta, alpha, num_iters)

    plotJ(J_history, num_iters)

    return mu, sigma, theta

def loadtxtAndcsv_data(fileName, split, dataType):
    return np.loadtxt(fileName, delimiter = split, dtype = dataType)

def loadtxtAndcsv_datadnpy_data(fileName):
    return np.load(fileName)

def featureNormaliza(X):
    X_norm = np.array(X)
    print(X_norm)
    mu = np.zeros((1, X.shape[1]))
    sigma = np.zeros((1, X.shape[1]))

    mu = np.mean(X_norm, 0)
    sigma = np.std(X_norm, 0)
    for i in range(X.shape[1]):
        X_norm[:, i] = (X_norm[:, i] - mu[i]) / sigma[i]

    print(X_norm)
    return X_norm, mu, sigma

def plot_X1_X2(X):
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    n = len(theta)

    temp = np.matrix(np.zeros((n, num_iters)))

    J_history = np.zeros((num_iters, 1))

    for i in range(num_iters):
        h = np.dot(X, theta)
        temp[:, i] = theta -((alpha / m) * (np.dot(np.transpose(X), h -y)))
        theta = temp[:, i]
        J_history[i] = computerCost(X, y, theta)
        print('.', end= ' ')

    return theta, J_history

def computerCost(X, y, theta):
    m = len(y)
    J = 0

    J = (np.transpose(X * theta - y)) * (X * theta - y) / (2 * m)
    return J

def plotJ(J_history, num_iters):
    x = np.arange(1, num_iters + 1)
    plt.plot(x, J_history)
    plt.xlabel(u"迭代次数", fontproperties = font)
    plt.ylabel(u"代价值", fontproperties = font)
    plt.title(u"代价值随迭代次数的变化", fontproperties = font)
    plt.show()


def testLinearRegression():
    mu, sigma, theta, = linearRegression(0.05, 1000)


def predict(mu, sigma, theta):
    result = 0
    predict = np.array([1650, 3])
    norm_predict = (predict - mu) / sigma
    final_predict = np.hstack((np.ones((1)), norm_predict))

    result = np.dot(final_predict, theta)
    return result

if __name__ == "__main__":
    testLinearRegression()
    
