# version 1: without investigation of H and xGrid
import autograd.numpy as np
# from autograd import grad 
from autograd import elementwise_grad
import autograd.numpy.random as npr
from autograd.core import primitive
from matplotlib import pyplot as plt
import time
import os.path

def partOne(x):
    p1 = x + (1. + 3.*x**2) / (1. + x + x**3)
    return p1

def partTwo(x):
    p2 = x**3 + 2.*x + x**2 * ((1. + 3.*x**2) / (1. + x + x**3))
    return p2

def f(x, psy):
    fx = partTwo(x) - partOne(x) * psy
    return fx

def analyticSol(x):
    sol = (np.exp((-x**2)/2.)) / (1. + x + x**3) + x**2
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
    N = np.dot(a1, W[1])    # do we need bias for hidden layer to output layer? + b[1] 
    return N

# derivative of neural network; 1th order
def NNDerivative(W, x):
    VW = np.dot(W[1].T, W[0].T)
    z = np.dot(x, W[0]) + W[2]
    NNDeriv = np.dot(VW, sigFirstDerivative(z))  # sigFirstDerivative(z); z = np.dot(x, W[0]) # do we need +b[1]?
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
    sol = 1 + x * NN(W, x)[0][0]
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

# relative error norm; k=1: max norm, otherwise: euclidean norm
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


# start counting wall clock time for traning NN
startWall = time.time()
# start counting CPU clock time for traning NN
startCPU = time.process_time()


# number of discretization points
xGrid = 20
X = np.linspace(0, 1, xGrid)
# W = [npr.randn(1, 10), npr.randn(10, 1)]

# Number of hidden nodes
H = 20
W = [npr.randn(1, H), npr.randn(H, 1), npr.randn(1, H)]  # bias: W[2]
# b = npr.randn(1, 10)    # two biases? [npr.randn(1, 10), npr.randn(10, 1)] --- only one biase! 1 biase for 1 sigmoid function
# leaning rate lambda
lmb = 0.001
itr = 1000


#Amin: change the following into a while loop
# the stopping condition should have two components:
# 1. i becoming larger than a max_iteration parameter
# 2. rel error of the i-th iteration going below a tolerance (call it tol, e.g, 0.1)
# So, you would need to calculate the rel error at each iteration

# calculate partial derivative and update W
tol = 0.1
idx = 0
relErrMax = np.zeros(itr)
relErrMax[idx] = errorEval(funcDiff, 0, 1, 100, W, 1)

# calculate partial derivative and update W
while not (relErrMax[idx] < tol) and idx < itr - 1:
    # costGrad = grad(costFunction)(W, X, 1)
    costGrad = elementwise_grad(costFunction, 0)(W, X, 1)

    W[0] = W[0] - lmb * costGrad[0]
    W[1] = W[1] - lmb * costGrad[1]
    W[2] = W[2] - lmb * costGrad[2]
    
    idx += 1
    relErrMax[idx] = errorEval(funcDiff, 0, 1, 100, W, 1)


print("relative error for max norm at the final iteration:", relErrMax[idx])
print("iteration count:", idx + 1)

# end counting CPU clock time for training NN
endCPU = time.process_time()
cpuTime = endCPU - startCPU
# end counting wall clock time for traning NN
endWall = time.time()
wallTime = endWall - startWall

print("weight from input unit to hidden unit:", W[0])
print("weight from hidden unit to the output:", W[1])
print("bias:", W[2])
print("cost value:", costFunction(W, X, 1))
print("wall time for training NN:", wallTime)
print("CPU time for training NN:", cpuTime)


# For plotting, we can use as many points as we like
numPlotPoints = 100
XPlot = np.linspace(0, 1, numPlotPoints)

ySol = analyticSol(XPlot)
yRes = [1 + xi*NN(W, xi)[0][0] for xi in XPlot] 


# record results in text files
filePath = 'C:/Users/DELL/Desktop/DT/codes/records'
now = time.strftime("%m%d%H%M")
fileName = 'record_' + now + '.txt'
fileDir = os.path.join(filePath, fileName)

file = open(fileDir, 'w')

file.write('The number of discretization points: ' + str(xGrid) +
            '\nThe number of hidden nodes: ' + str(H) +
            '\nLearning rate: ' + str(lmb) +
            '\nIteration number: ' + str(itr) +
            '\nThe weight from input unit to hidden unit: ' + str(W[0]) +
            '\nThe weight from hidden unit to the output: ' + str(W[1]) +
            '\nThe bias: ' + str(W[2]) +
            '\nCost value: ' + str(costFunction(W, X, 1)) + 
            '\nWall time for training NN: ' + str(wallTime) +
            '\nCPU time for training NN: ' + str(cpuTime) +
            '\nThe number of plotting points: ' + str(numPlotPoints) +
            '\nRelative error for max norm: ' + str(relErrMax))

file.close()


# plot analytic solution and nn-trained result
plt.figure()
plt.plot(XPlot, ySol, color = 'blue') 
plt.plot(XPlot, yRes, color = 'red')
plt.show()

# plot relative errors w.r.t. each iteration
itrPlot = np.linspace(1, itr, itr)

plt.figure()
plt.plot(itrPlot, relErrMax, color = 'green')
plt.show()


# TODO:
#   1) Measure time, both wall clock and CPU clock  --- for training NN? or for the whole program?
#   2) Investigate how to modify optimization for different cost functions (Not all cost functions can be minimized by 1000 iterations 
#   and the same learning rate) 
#   Read paper: super-convergance: very fast training of NN using large learning rates --- hard to understand HF
#   3) Store the results to a file (so that you can compare them against each other)
#   4) Investigate how the accuracy is related to the number of hidden nodes, and also the number of discretization points
