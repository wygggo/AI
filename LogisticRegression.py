#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from matplotlib.font_manager import FontProperties
font = FontProperties(fname = r"/home/AI/mlpractise/font/simsun.ttc", size = 14)

def LogisticRegression():
    data = loadtxtAndcsv_data("data2.txt", ",", np.float64)
    X = data[:,0:-1]
    y = data[:,-1]
    
    plot_data(X, y)

    X = mapFeature(X[:, 0], X[:,1])
    initial_theta = np.zeros((X.shape[1],1))
    initial_lambda = 0.1 #初始化正则系数

    J = costFunction(initial_theta, X, y, initial_lambda)

    print(J)
    result = optimize.fmin_bfgs(costFunction, initial_theta, fprime = gradient, args = (X, y, initial_lambda))
    p = predict(X, result)
    print(u'在训练集上的精确度为%f%%' % np.mean(np.float64(p == y) * 100))

    X = data[:, 0: -1]
    y = data[:, -1]
    plotDecisionBoundary(result, X, y)


def loadtxtAndcsv_data(fileName, split, dataType):
    return np.loadtxt(fileName, delimiter = split, dtype = dataType)


def loadnpy_data(fileName):
    return np.load(fileName)

def plot_data(X, y):
    pos = np.where(y == 1)
    neg = np.where(y == 0)

    plt.figure(figsize = (15, 12))
    plt.plot(X[pos, 0], X[pos, 1], 'ro')
    plt.plot(X[neg, 0], X[neg, 1], 'bo')
    plt.title(u"两个类别散点图", fontproperties = font)
    plt.show()


def mapFeature(X1, X2):
    degree = 2;
    out = np.ones((X1.shape[0], 1))

    for i in np.arange(1, degree + 1):
        for j in range(i + 1):
            temp = X1 ** (i - j) * (X2 ** j)
            out = np.hstack((out, temp.reshape(-1, 1)))

    return out

def costFunction(initial_theta, X, y, initial_lambda):
    m = len(y)
    J = 0

    h = sigmoid(np.dot(X, initial_theta))
    theta1 = initial_theta.copy()
    theta1[0] = 0

    temp = np.dot(np.transpose(theta1), theta1)
    J = (-np.dot(np.transpose(y), np.log(h)) - np.dot(np.transpose(1 - y), np.log(1 - h)) + temp * initial_lambda / 2) / m
    return J

def gradient(initial_theta, X, y, initial_lambda):
    m = len(y)
    grad = np.zeros((initial_theta.shape[0]))

    h =  sigmoid(np.dot(X, initial_theta))
    theta1 = initial_theta.copy()
    theta1[0] = 0

    grad = np.dot(np.transpose(X), h - y) / m + initial_lambda / m * theta1
    return grad

def sigmoid(z):
    h = np.zeros((len(z), 1))
    h = 1.0 / (1.0 + np.exp(-z))
    return h

def plotDecisionBoundary(theta, X, y):
    pos = np.where(y == 1)
    neg = np.where(y == 0)

    plt.figure(figsize=(15, 12))
    plt.plot(X[pos, 0], X[pos, 1], 'ro')
    plt.plot(X[neg, 0], X[neg, 1], 'bo')
    plt.title(u"决策边界", fontproperties = font)

    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)

    z = np.zeros((len(u), len(v)))
    
    for i in range(len(u)):
        for j in range(len(v)):
            z[i, j] = np.dot(mapFeature(u[i].reshape(1, -1), v[j].reshape(1, -1)), theta)

    z = np.transpose(z)
    plt.contour(u, v, z, [0, 0.01], linewidth = 2.0)
    plt.show()

def predict(X, theta):
    m = X.shape[0]
    p = np.zeros((m, 1))
    p = sigmoid(np.dot(X, theta))

    for i in range(m):
        if p[i] > 0.5:
            p[i] = 1
        else:
            p[i] = 0

    return p

def testLogisticRegression():
    LogisticRegression()

if __name__ == "__main__":
    testLogisticRegression()
    
