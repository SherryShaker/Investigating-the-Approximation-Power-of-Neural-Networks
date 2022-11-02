# parametric ODE version 2: Initial Value Problem - first order
# solve y'(t) = y(t), y(t) = Ae^t
import autograd.numpy as np
from autograd import grad 
import autograd.numpy.random as npr
from autograd.core import primitive
from matplotlib import pyplot as plt
import time
import os.path


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

def f(x, y):
    fxy = y 
    return fxy

# cost function for a trial solution of the first derivative ODE
def costFunction(W, X, A, x0):
    cost = 0.
    for xi in X:
        for aj in A:
            nnOut = NN(W, xi)[0][0]
            psy = aj + (xi - x0) * nnOut

            nnDerivOut = NNDerivative(W, xi)[0][0]
            psyDeriv = nnOut + (xi - x0) * nnDerivOut

            #func = f(xi, aj)       
            func = f(xi, psy)       
            errSqr = (psyDeriv - func)**2

            cost += errSqr

    return cost

def analyticSol(x, a):
    sol = a * np.exp(x)
    return sol

# function expression for result solution
def resultSol(W, x, a):
    sol = a + x * NN(W, x)[0][0]
    return sol

# calculate the difference between result solution and analytic solution
def funcDiff(W, x, a):
    yRes = resultSol(W, x, a)
    ySol = analyticSol(x, a)
    diff = yRes - ySol
    return diff

# calculate max norm for two functions within boundaries
def maxNorm(a, b, m, n, numOfPoints, W):
    maxVal = 0
    X = np.linspace(a, b, numOfPoints)
    A = np.linspace(m, n, numOfPoints)
    for x in X:
        for a in A:
            maxVal = max(maxVal, np.abs(funcDiff(W, x, a)))
    return maxVal

# calculate max value for a function within boundaries    
def maxFunc(a, b, m, n, numOfPoints):
    maxVal = 0
    X = np.linspace(a, b, numOfPoints)
    A = np.linspace(m, n, numOfPoints)
    for x in X:
        for a in A:
            maxVal = max(maxVal, np.abs(analyticSol(x, a)))
    return maxVal

# calculate euclidean norm for two functions within boundaries
def euclideanNorm(a, b, m, n, numOfPoints, W):
    culVal = 0
    X = np.linspace(a, b, numOfPoints)
    A = np.linspace(m, n, numOfPoints)
    for x in X:
        for a in A:
            diff = np.abs(funcDiff(W, x, a))
            culVal += diff**2
    eucliVal = np.sqrt(culVal)
    return eucliVal

# calculate euclidean value for a function within boundaries
def euclideanFunc(a, b, m, n, numOfPoints):
    culVal = 0
    X = np.linspace(a, b, numOfPoints)
    A = np.linspace(m, n, numOfPoints)
    for x in X:
        for a in A:
            culVal += analyticSol(x, a)**2
    eucliVal = np.sqrt(culVal)
    return eucliVal

# relative error norm; k=1: max norm, otherwise: euclidean norm
def errorEval(a, b, m, n, numOfPoints, W, k):
    if k == 1:
        maxN = maxNorm(a, b, m, n, numOfPoints, W)
        maxF = maxFunc(a, b, m, n, numOfPoints)
        relErrMax = maxN / maxF
        return relErrMax
    else:
        euclideanN = euclideanNorm(a, b, m, n, numOfPoints, W)
        euclideanF = euclideanFunc(a, b, m, n, numOfPoints)
        relErr2 = euclideanN / euclideanF
        return relErr2

# train neural network
def trainNN(lmb, itr, tol, H, X, A, W, x0):
    # count iteration times
    idx = 0

    # record relative error for Max Norm at each iteration
    relErrMax = np.zeros(itr)
    relErrMax[idx] = errorEval(0, 1, 0, 1, 100, W, 1)

    # start counting wall clock time for traning NN
    startWall = time.time()
    # start counting CPU clock time for traning NN
    startCPU = time.process_time()

    # calculate partial derivative and update W
    while (not (relErrMax[idx] < tol)) and (idx < itr - 1):
        costGrad = grad(costFunction,0)(W, X, A, x0)

        W[0] = W[0] - lmb * costGrad[0]
        W[1] = W[1] - lmb * costGrad[1]
        W[2] = W[2] - lmb * costGrad[2]
        
        idx += 1
        relErrMax[idx] = errorEval(0, 1, 0, 1, 100, W, 1)

    # end counting CPU clock time for training NN
    endCPU = time.process_time()
    cpuTime = endCPU - startCPU
    # end counting wall clock time for traning NN
    endWall = time.time()
    wallTime = endWall - startWall

    return W, cpuTime, wallTime, relErrMax, idx


# discretization points for parameter x
xGrid = 10
X = np.linspace(0, 1, xGrid)
# discretization points for parameter A
aGrid = 10
A = np.linspace(0, 1, aGrid)
# number of hidden nodes
H = 10
# initialize weight and bias: w: W[0], v: W[1], b: W[2]
W = [npr.randn(1, H), npr.randn(H, 1), npr.randn(1, H)]
# leaning rate lambda
lmb = 0.001
# iteration setting
itr = 1000
# tolerence rate setting
tol = 0.1

# train a neural network w.r.t. parameter X and A
W, cpuTime, wallTime, relErrMax, itrTimes = trainNN(lmb, itr, tol, H, X, A, W, 0)

print("weight from input unit to hidden unit:", W[0])
print("weight from hidden unit to the output:", W[1])
print("bias:", W[2])
print("cost value:", costFunction(W, X, A, 0))
print("wall time for training NN:", wallTime)
print("CPU time for training NN:", cpuTime)
print("relative error for max norm at the final iteration:", relErrMax[itrTimes])
print("iteration count:", itrTimes + 1)


# For plotting, we can use as many points as we like
numPlotPoints = 100

# record results in text files
filePath = 'C:/Users/DELL/Desktop/DT/codes/records'
now = time.strftime("%m%d%H%M")
fileName = 'record_' + now + '.txt'
fileDir = os.path.join(filePath, fileName)

file = open(fileDir, 'w')

file.write('The number of discretization points for parameter X: ' + str(xGrid) +
            '\nThe number of discretization points for parameter A: ' + str(aGrid) +
            '\nThe number of hidden nodes: ' + str(H) +
            '\nLearning rate: ' + str(lmb) +
            '\nSetting Iteration number: ' + str(itr) +
            '\nThe weight from input unit to hidden unit: ' + str(W[0]) +
            '\nThe weight from hidden unit to the output: ' + str(W[1]) +
            '\nThe bias: ' + str(W[2]) +
            '\nCost value: ' + str(costFunction(W, X, A, 0)) + 
            '\nWall time for training NN: ' + str(wallTime) +
            '\nCPU time for training NN: ' + str(cpuTime) +
            '\nThe number of plotting points: ' + str(numPlotPoints**2) +
            '\nRelative error for max norm: ' + str(relErrMax) +
            '\nIteration Times: ' + str(itrTimes+1))

file.close()


# plot relative errors w.r.t. each iteration
itrPlot = np.linspace(1, itr, itr)

plt.figure()
plt.plot(itrPlot, relErrMax, color = 'green')
plt.savefig('C:/Users/DELL/Desktop/DT/codes/records/fig_relErr.png')
plt.show()

# plot 3d figure of each H and xGrid with corresponding relative error
# prepare arrays x: H, y: xGrid, z: relErr
X= np.linspace(0, 1, numPlotPoints)
Y = X.copy().T
X, Y = np.meshgrid(X, Y)
Z = [Y + X*NN(W, x.reshape(numPlotPoints,1))[0][0] for x in X][0]

x = np.linspace(0, 1, numPlotPoints)
y = x.copy().T
x, y = np.meshgrid(x, y)
z = analyticSol(x, y)

# plot figure 1
fig = plt.figure(figsize =(10, 6)) 
ax = fig.add_subplot(1,2,1, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap = plt.get_cmap('rainbow'), edgecolor ='none') 
# plot figure 2 
ax2 = fig.add_subplot(1,2,2, projection='3d')
surf2 = ax2.plot_surface(x, y, z, cmap = plt.get_cmap('rainbow'), edgecolor ='none')

ax.set_xlabel('X')
ax.set_ylabel('A')
ax.set_zlabel('resultSol')

ax2.set_xlabel('x')
ax2.set_zlabel('analyticSol')

fig.colorbar(surf, shrink=0.5, aspect=10)
plt.savefig('C:/Users/DELL/Desktop/DT/codes/records/fig_3d.png')
# show plot 
plt.show()
