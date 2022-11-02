import autograd.numpy as np
from autograd import grad 
import autograd.numpy.random as npr
from autograd.core import primitive
from matplotlib import pyplot as plt

def partOne(x):
    p1 = 1. / 5.
    return p1

def partTwo(x):
    p2 = np.exp(-x/5.) * np.cos(x)
    return p2

def f(x, psy):
    fx = partTwo(x) - partOne(x) * psy
    return fx

def analyticSol(x):
    sol = np.exp(-x/5.) * np.sin(x)
    return sol

def sigmoid(x):
	sigFunc = 1 / (1 + np.exp(-x))
	return sigFunc

# first order derivative of sigmoid function
def sigFirstDerivative(x):
    firstD = sigmoid(x) * (1 - sigmoid(x))
    return firstD

def NN(W, x):
    z = np.dot(x, W[0]) + W[2]
    a1 = sigmoid(z)
    N = np.dot(a1, W[1])
    return N

# derivative of neural network; 1th order
def NNDerivative(W, x):
    VW = np.dot(W[1].T, W[0].T)
    z = np.dot(x, W[0]) + W[2]
    NNDeriv = np.dot(VW, sigFirstDerivative(z))  # sigFirstDerivative(z); z = np.dot(x, W[0])
    return NNDeriv

# cost function for a trial solution of the first derivative ODE
def costFunction(W, x, ic):
    A = ic
    cost = 0.
    for xi in x:
        nnOut = NN(W, xi)[0][0]
        psy = A + xi * nnOut

        nnDerivOut = NNDerivative(W, xi)[0][0]
        psyDeriv = nnOut + xi * nnDerivOut

        func = f(xi, psy)       
        errSqr = (psyDeriv - func)**2

        cost += errSqr
    return cost

# function expression for result solution
def resultSol(W, x):
    sol = x * NN(W, x)[0][0]
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
    costGrad =  grad(costFunction)(W, X, 0)

    W[0] = W[0] - lmb * costGrad[0]
    W[1] = W[1] - lmb * costGrad[1]
    W[2] = W[2] - lmb * costGrad[2]

print("weight from input unit to hidden unit:", W[0])
print("weight from hidden unit to the output:", W[1])
print("bias:", W[2])
print("cost value:", costFunction(W, X, 0))


# For plotting, we can use as many points as we like
numPlotPoints = 100
XPlot = np.linspace(0, 1, numPlotPoints)

ySol = analyticSol(XPlot)
yRes = [xi*NN(W, xi)[0][0] for xi in XPlot] 

# calculate real error max norm
relErrMax = errorEval(funcDiff, 0, 2, 100, W, 1)
print("relative error for max norm:", relErrMax)

# plot analytic solution and nn-trained result
plt.figure()
plt.plot(XPlot, ySol, color = 'blue') 
plt.plot(XPlot, yRes, color = 'red')
plt.show()