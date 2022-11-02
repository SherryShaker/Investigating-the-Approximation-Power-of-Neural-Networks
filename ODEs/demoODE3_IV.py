import autograd.numpy as np
from autograd import grad 
import autograd.numpy.random as npr
from autograd.core import primitive
from matplotlib import pyplot as plt

def partOne(x, psyD, psy):
    p1 = 1. / 5. * psyD + psy
    return p1

def partTwo(x):
    p2 = - 1. / 5. * np.exp(-x/5.) * np.cos(x)
    return p2

def f(x, psyD, psy):
    fx = partTwo(x) - partOne(x, psyD, psy)
    return fx

def analyticSol(x):
    sol = np.exp(-x/5.) * np.sin(x)
    return sol

def sigmoid(x):
	sigFunc = 1 / (1 + np.exp(-x))
	return sigFunc

# first order derivative of sigmoid function
def sig1Derivative(x):
    firstD = sigmoid(x) * (1 - sigmoid(x))
    return firstD

# second order derivative of sigmoid function
def sig2Derivative(x):
    secondD = sig1Derivative(x) * (1 - 2 * sigmoid(x))
    return secondD

def NN(W, x):
    z = np.dot(x, W[0]) + W[2]
    a1 = sigmoid(z)
    N = np.dot(a1, W[1])
    return N

# 1th order derivative of neural network
def NN1Derivative(W, x):
    z = np.dot(x, W[0]) + W[2]
    VW = np.dot(W[1].T, W[0].T)
    NNDeriv = np.dot(VW, sig1Derivative(z))  # sigFirstDerivative(z); z = np.dot(x, W[0])
    return NNDeriv

# 2ed order derivative of neural network
def NN2Derivative(W, x):
    z = np.dot(x, W[0]) + W[2]
    VW = np.dot(W[1].T, W[0].T**2)
    NNDeriv = np.dot(VW, sig2Derivative(z))  # sigFirstDerivative(z); z = np.dot(x, W[0])
    return NNDeriv

# cost function for a trial solution of the first derivative ODE
def costFunction(W, x, ic1, ic2):
    A1 = ic1
    A2 = ic2
    cost = 0.
    for xi in x:
        nnOut = NN(W, xi)[0][0]
        nn1DerivOut = NN1Derivative(W, xi)[0][0]
        nn2DerivOut = NN2Derivative(W, xi)[0][0]
        
        psy = A1 + A2 * xi + xi**2 * nnOut
        psy1Deriv = A2 + 2 * xi * nnOut + xi**2 * nn1DerivOut
        psy2Deriv = 2 * nnOut + 4 * xi * nn1DerivOut + xi**2 * nn2DerivOut
        # psy1Deriv = nnOut + xi * nn1DerivOut
        # psy2Deriv = 2. * nn1DerivOut + xi * nn2DerivOut

        func = f(xi, psy1Deriv, psy)       
        errSqr = (psy2Deriv - func)**2

        cost += errSqr
    return cost

# function expression for result solution
def resultSol(W, x):
    sol = x + x**2 * NN(W, x)[0][0]
    return sol

# calculate the difference between result solution and analytic solution
def funcDiff(W, x):
    yRes = resultSol(W, x)
    ySol = analyticSol(x)
    diff = yRes - ySol
    return diff

# calculate max norm for two functions within boundaries
def maxNorm(funcDiff, a, b, numOfPoints, W):
    maxVal = 0
    X = np.linspace(a, b, numOfPoints)
    for x in X:
        maxVal = max(maxVal, np.abs(funcDiff(W, x)))
    return maxVal

# calculate max value for a function within boundaries    
def maxFunc(a, b, numOfPoints):
    maxVal = 0
    X = np.linspace(a, b, numOfPoints)
    for x in X:
        maxVal = max(maxVal, np.abs(analyticSol(x)))
    return maxVal

# calculate euclidean norm for two functions within boundaries
def euclideanNorm(funcDiff, a, b, numOfPoints, W):
    culVal = 0
    X = np.linspace(a, b, numOfPoints)
    for x in X:
        diff = np.abs(funcDiff(W, x))
        culVal += diff**2
    eucliVal = np.sqrt(culVal)
    return eucliVal

# calculate euclidean value for a function within boundaries
def euclideanFunc(a, b, numOfPoints):
    culVal = 0
    X = np.linspace(a, b, numOfPoints)
    for x in X:
        culVal += analyticSol(x)**2
    eucliVal = np.sqrt(culVal)
    return eucliVal

# real error norm; k=1: max norm, otherwise: euclidean norm
def errorEval(funcDiff, a, b, numOfPoints, W, k):
    if k == 1:
        maxN = maxNorm(funcDiff, a, b, numOfPoints, W)
        maxF = maxFunc(a, b, numOfPoints)
        relErrMax = maxN / maxF
        return relErrMax
    else:
        euclideanN = euclideanNorm(funcDiff, a, b, numOfPoints, W)
        euclideanF = euclideanFunc(a, b, numOfPoints)
        relErr2 = euclideanN / euclideanF
        return relErr2


xGrid = 10
X = np.linspace(0, 2, xGrid)
# W = [npr.randn(1, 10), npr.randn(10, 1)]
W = [npr.randn(1, 10), npr.randn(10, 1), npr.randn(1, 10)]  # bias: W[2]
lmb = 0.001

# calculate partical derivative and update W
for i in range(1000):
    costGrad =  grad(costFunction)(W, X, 0, 1)

    W[0] = W[0] - lmb * costGrad[0]
    W[1] = W[1] - lmb * costGrad[1]
    W[2] = W[2] - lmb * costGrad[2]

print("weight from input unit to hidden unit:", W[0])
print("weight from hidden unit to the output:", W[1])
print("bias:", W[2])
print("cost value:", costFunction(W, X, 0, 1))


# For plotting, we can use as many points as we like
numPlotPoints = 100
XPlot = np.linspace(0, 1, numPlotPoints)

ySol = analyticSol(XPlot)
yRes = [xi + xi**2*NN(W, xi)[0][0] for xi in XPlot] 

# calculate real error max norm
relErrMax = errorEval(funcDiff, 0, 2, 100, W, 1)
print("relative error for max norm:", relErrMax)

# plot analytic solution and nn-trained result
plt.figure()
plt.plot(XPlot, ySol, color = 'blue') 
plt.plot(XPlot, yRes, color = 'red')
plt.show()