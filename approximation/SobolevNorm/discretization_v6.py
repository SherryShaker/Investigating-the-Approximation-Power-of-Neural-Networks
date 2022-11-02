# discretization points method version 1
# u(x, y) = 10x(1-x)y(1-y)(1-2y)
# u' w.r.t. x = 10(1-2x)y(1-y)(1-2y); u' w.r.t. y = 10x(1-x)(6y^2-6y+1)
# implement symbolic gradient descent 
import autograd.numpy as np
from autograd import grad 
import autograd.numpy.random as npr
from autograd.core import primitive
from matplotlib import pyplot as plt
import time
import os.path


def ReLU(x):
    result = np.where(x > 0, x, 0)
    return result

# first order derivative of ReLU
def ReLUFirstDerivative(x):
    dx = np.ones_like(x)
    dx[x <= 0] = 0
    return dx

def NN(W, x, y):
    XY = np.asarray([x, y])
    z = np.dot(XY, W[0]) + W[2]
    a = ReLU(z)
    N = np.dot(a, W[1])
    return N

# 1st derivative of NN
def NNDerivative(W, x, y):
    XY = np.asarray([x, y])
    VW = W[1].T * W[0]
    z = np.dot(XY, W[0]) + W[2]
    NNDeriv = np.dot(VW, ReLUFirstDerivative(z).T)

    NNDeriv_x = NNDeriv[0][0]
    NNDeriv_y = NNDeriv[1][0]
    result = np.asarray([NNDeriv_x, NNDeriv_y])
    return result

# the given function
def U(x, y):
    u = 10 * x * (1 - x) * y * (1 - y) * (1 - 2*y)
    return u

# the derivative of the given function
def UDerivative(x, y):
    u_x = 10 * (1 - 2*x) * y * (1 - y) * (1 - 2*y)
    u_y = 10 * x * (1 - x) * (6 * y**2 - 6 * y + 1)
    uDerive = np.asarray([u_x, u_y]) 
    return uDerive

# cost function for the known function
def costFunction(W, T, S):
    # error square sum of points inside the domain
    cost1 = 0.
    for ti in T:
        xi, yi = ti
        # the error sum of NN
        nnOut = NN(W, xi, yi)[0][0]
        fOut = U(xi, yi)
        errSqr = (nnOut - fOut)**2
        cost1 += errSqr
        #  the error sum of NN derivative 
        nnDerivOut = NNDerivative(W, xi, yi)
        uDerive = UDerivative(xi, yi)
        res = uDerive - nnDerivOut
        errSqr_deriv = res[0]**2 + res[1]**2
        cost1 += errSqr_deriv
    # error square sum of points on the boundary
    cost2 = 0.
    for si in S:
        xi, yi = si
        # the error sum of NN
        nnOut = NN(W, xi, yi)[0][0]
        fOut = U(xi, yi)
        errSqr = (nnOut - fOut)**2
        cost2 += errSqr
        #  the error sum of NN derivative 
        nnDerivOut = NNDerivative(W, xi, yi)
        uDerive = UDerivative(xi, yi)
        res = uDerive - nnDerivOut
        errSqr_deriv = res[0]**2 + res[1]**2
        cost2 += errSqr_deriv
    # cost for the trained NN
    cost = 0.5 * (cost1 + cost2)
    return cost

def difference_NN_U(W, x, y):
    nnOut = NN(W, x, y)[0][0]
    fOut = U(x, y)
    diff = nnOut - fOut
    return diff

def partial_derivative_W_helper(W, xi, yi):
    w = W[0]
    v = W[1]
    b = W[2]
    # cost grad initilization
    cost_deriv = np.zeros(w.shape, dtype=np.float)
    # define input values
    inputs = np.asarray([[xi, yi]])
    x = np.asarray([xi, 0])
    y = np.asarray([0, yi])
    # z with respect to inputs x and y
    z = np.dot(inputs, w) + b
    # z with respect to input x or y
    z_x = np.dot(x, w) + b
    z_y = np.dot(y, w) + b

    # difference between NN result and U result
    diff = difference_NN_U(W, xi, yi)
    # matrix multiplication result
    matrix = v * ReLUFirstDerivative(z).T * inputs
    # error sum with respect to NN and original function (U)
    cost_deriv += diff * matrix.T

    # difference between NN derivative and U derivative w.r.t. input x or y
    nnDerivOut = NNDerivative(W, xi, yi)
    uDerive = UDerivative(xi, yi)
    diff_x = nnDerivOut[0] - uDerive[0]
    diff_y = nnDerivOut[1] - uDerive[1]
    # matrix multiplication result w.r.t. x or y
    matrix_x = diff_x * v.T * ReLUFirstDerivative(z_x)
    matrix_y = diff_y * v.T * ReLUFirstDerivative(z_y)
    # error sum with respect to the Laplace of NN and original function (U)
    cost_deriv += np.asarray([matrix_x[0], matrix_y[0]])
    return cost_deriv

# cost derivative with respect to weight (W[0])
def partial_derivative_W(W, T, S):
    # cost1: costs grad related to points in set T (within the boundary)
    cost_deriv_1 = np.zeros(W[0].shape, dtype=np.float)
    for ti in T:
        xi, yi = ti
        cost_deriv_1 += partial_derivative_W_helper(W, xi, yi)
    # cost2: costs grad related to points in set S (on the boundary)
    cost_deriv_2 = np.zeros(W[0].shape, dtype=np.float)
    for si in S:
        xi, yi = si
        cost_deriv_2 += partial_derivative_W_helper(W, xi, yi)
    result = cost_deriv_1 + cost_deriv_2
    return result

def partial_derivative_V_helper(W, xi, yi):
    w = W[0]
    v = W[1]
    b = W[2]
    # weights w.r.t. input x
    w_x = W[0][0]
    # weights w.r.t. input y
    w_y = W[0][1]
    # cost derivative initilization
    cost_deriv = np.zeros(v.shape, dtype=np.float)
    # define input values
    inputs = np.asarray([xi, yi])
    x = np.asarray([xi, 0])
    y = np.asarray([0, yi])

    # z with respect to inputs x and y
    z = np.dot(inputs, w) + b
    # z with respect to input x or y
    z_x = np.dot(x, w) + b
    z_y = np.dot(y, w) + b

    # error sum with respect to NN and original function (U)
    diff = difference_NN_U(W, xi, yi)
    cost_deriv += diff * ReLU(z).T

    # difference between NN derivative and U derivative
    nnDerivOut = NNDerivative(W, xi, yi)
    uDerive = UDerivative(xi, yi)
    diff_x = nnDerivOut[0] - uDerive[0]
    diff_y = nnDerivOut[1] - uDerive[1]
    # matrix multiplication result w.r.t. input x or y
    matrix_x = w_x * ReLUFirstDerivative(z_x)
    matrix_y = w_y * ReLUFirstDerivative(z_y)
    # error sum with respect to the Laplace of NN and original function (U)
    cost_deriv += diff_x * matrix_x.T + diff_y * matrix_y.T
    return cost_deriv

# cost derivative with respect to V (W[1])
def partial_derivative_V(W, T, S):
    # cost1: costs grad related to points in set T (within the boundary)
    cost_deriv_1 = np.zeros(W[1].shape, dtype=np.float)
    for ti in T:
        xi, yi = ti
        cost_deriv_1 += partial_derivative_V_helper(W, xi, yi)
    # cost2: costs grad related to points in set S (on the boundary)
    cost_deriv_2 = np.zeros(W[1].shape, dtype=np.float)
    for si in S:
        xi, yi = si
        cost_deriv_2 += partial_derivative_V_helper(W, xi, yi)
    result = cost_deriv_1 + cost_deriv_2
    return result

def partial_derivative_B_helper(W, xi, yi):
    w = W[0]
    v = W[1]
    b = W[2]
    inputs = np.asarray([xi, yi])
    z = np.dot(inputs, w) + b
    # matrix multiplication result
    matrix = v.T * ReLUFirstDerivative(z)
    # difference between NN result and U result
    diff = difference_NN_U(W, xi, yi)
    # error sum with respect to NN and original function (U)
    cost_deriv = 2 * diff * matrix
    return cost_deriv

# cost derivative with respect to bias (W[2])
def partial_derivative_B(W, T, S):
    # cost1: costs grad related to points in set T (within the boundary)
    cost_deriv_1 = np.zeros(W[2].shape, dtype=np.float)
    for ti in T:
        xi, yi = ti
        cost_deriv_1 += partial_derivative_B_helper(W, xi, yi)
    # cost2: costs grad related to points in set S (on the boundary)
    cost_deriv_2 = np.zeros(W[2].shape, dtype=np.float)
    for si in S:
        xi, yi = si
        cost_deriv_2 += partial_derivative_B_helper(W, xi, yi)
    result = cost_deriv_1 + cost_deriv_2
    return result

def sobolevNorm(W, points, k):
    # part 1 error: square root of error sum
    relErr1 = 0.
    for x, y in points:
        nnOut = NN(W, x, y)[0][0]
        u = U(x, y)
        # numerator of relative error: N - u
        if k == 1:
            f = nnOut - u
        # denominator of relative error: u
        else:
            f = u
        err = f**2
        relErr1 += err
    relErr1 = relErr1**0.5
    # part 2 error: square root of the derivative of error sum
    relErr2 = 0.
    for x, y in points:
        nnDerivOut = NNDerivative(W, x, y)
        uDerive = UDerivative(x, y)
        # nominator of relative error: N - u
        if k == 1:
            f = nnDerivOut - uDerive
        # denominator of relative error: u
        else:
            f = uDerive
        err = f[0]**2 + f[1]**2 
        relErr2 += err
    relErr2 = relErr2**0.5
    # relative error
    sobNorm = relErr1 + relErr2
    return sobNorm

# relative error: Sobolev Norm
# P.S. the area in the formula are cancelled 
def errorEval(W, points):
    sobolevNorm_1 = sobolevNorm(W, points, 1) 
    sobolevNorm_2 = sobolevNorm(W, points, 0) 
    relErr_sobolev = sobolevNorm_1 / sobolevNorm_2
    return relErr_sobolev

# create meshgrid coordinates
def generateCoordinates(xGrid, yGrid):
    # create  x and y axis
    X = np.linspace(0, 1, xGrid)
    Y = np.linspace(0, 1, yGrid)
    # generate meshgrid coordinates
    xv, yv = np.meshgrid(X, Y)
    xlist = xv.ravel()
    ylist = yv.ravel()
    coordinate = zip(xlist, ylist)
    # point coordinates list on the boundary
    S = []
    # point coordinates list within the boundary
    T = []
    for x, y in coordinate:
        point = (x, y)
        if x == 0 or x == 1 or y == 0 or y == 1:
            S.append(point)
        else:
            T.append(point)
    return S, T

# train neural network
def trainNN(lmb, itr, tol, H, W, xGrid, yGrid, eval_xGrid, eval_yGrid):
    # S: list of points on the boundry; T: list of points within the boundary
    S, T = generateCoordinates(xGrid, yGrid)
    # chosen points coordinate list
    s_list, t_list = generateCoordinates(eval_xGrid, eval_yGrid)
    points = s_list + t_list
    # count iteration times
    idx = 0
    # record cost list
    costList = np.zeros(itr)
    costList[idx] = costFunction(W, T, S)
    # record relative error for Max Norm at each iteration
    relErr = np.zeros(itr)
    relErr[idx] = errorEval(W, points)
    # print("relErr", relErr[idx])
    # start counting wall clock time for traning NN
    startWall = time.time()
    # start counting CPU clock time for traning NN
    startCPU = time.process_time()

    # calculate partial derivative and update W
    while (not (relErr[idx] < tol)) and (idx < itr - 1):
        # print("w", W)
        costGrad_W = partial_derivative_W(W, T, S)
        costGrad_V = partial_derivative_V(W, T, S)
        costGrad_B = partial_derivative_B(W, T, S)
        # print("grad W", costGrad_W)
        # print("grad V", costGrad_V)
        # print("grad B", costGrad_B)

        W[0] = W[0] - lmb * costGrad_W
        W[1] = W[1] - lmb * costGrad_V
        W[2] = W[2] - lmb * costGrad_B

        # print("w", W)
        
        idx += 1
        costList[idx] = costFunction(W, T, S)
        relErr[idx] = errorEval(W, points)
        print("relErr", relErr[idx])
        print("cost", costList[idx])

    # end counting CPU clock time for training NN
    endCPU = time.process_time()
    cpuTime = endCPU - startCPU
    # end counting wall clock time for traning NN
    endWall = time.time()
    wallTime = endWall - startWall

    return W, cpuTime, wallTime, relErr, idx, costList


# number of points within the boundary
M = 64 #3364
# number of points on the boundary
K = 36 #1196
#  number of grid points in x axis
xGrid = 10 # 300
#  number of grid points in y axis
yGrid = 10 # 300
# number of evaluation grid point in x axis
eval_xGrid = 20
# number of evaluation grid point in x axis
eval_yGrid = 20
# number of hidden nodes
H = 10
# initialize weight and bias: w: W[0], v: W[1], b: W[2]
W = [npr.randn(2, H), npr.randn(H, 1), npr.randn(1, H)]
# leaning rate lambda
lmb = 0.0001
# iteratio n setting
itr = 1000
# tolerence rate setting
tol = 0.05

# train a neural network w.r.t. parameter X and A
W, cpuTime, wallTime, relErr, itrTimes, costList = trainNN(lmb, itr, tol, H, W, xGrid, yGrid, eval_xGrid, eval_yGrid)

# print records
print("weight from input unit to hidden unit:", W[0])
print("weight from hidden unit to the output:", W[1])
print("bias:", W[2])
print("wall time for training NN:", wallTime)
print("CPU time for training NN:", cpuTime)
print("iteration count:", itrTimes + 1)
print("cost value at the final iteration:", costList[itrTimes])
print("relative error for max norm at the final iteration:", relErr[itrTimes])


# record results in text files
current_path = os.path.abspath(__file__)
filePath = os.path.join(os.path.abspath(os.path.dirname(current_path) + os.path.sep + "."),'records')
# filePath = 'C:/Users/DELL/Desktop/DT/codes/records'
now = time.strftime("%m%d%H%M")
fileName = 'record_' + now + '.txt'
fileDir = os.path.join(filePath, fileName)

file = open(fileDir, 'w')

file.write('The number of points within the boundary M:' + str(M) +
            '\nThe number of points on the boundary K:' + str(K) +
            '\nThe number of grid points in x axis: ' + str(xGrid) +
            '\nThe number of grid points in y axis: ' + str(yGrid) +
            '\nThe number of hidden nodes: ' + str(H) +
            '\nLearning rate: ' + str(lmb) +
            '\nSetting Iteration number: ' + str(itr) +
            '\nThe weight from input unit to hidden unit: ' + str(W[0]) +
            '\nThe weight from hidden unit to the output: ' + str(W[1]) +
            '\nThe bias: ' + str(W[2]) +
            '\nWall time for training NN: ' + str(wallTime) +
            '\nCPU time for training NN: ' + str(cpuTime) +
            '\nIteration Times: ' + str(itrTimes + 1) +
            '\nCost value list: ' + str(costList) + 
            '\nRelative error for Soboler Norm: ' + str(relErr))

file.close()

# plot relative errors w.r.t. each iteration
figPath = os.path.join(os.path.abspath(os.path.dirname(current_path) + os.path.sep + "."),'records')
itrPlot = np.linspace(1, itr, itr)

fig1_name = 'relative_error_' + str(xGrid) + '.png'
fig1_save = os.path.join(figPath, fig1_name)

plt.figure()
plt.plot(itrPlot, relErr, color = 'blue')
plt.savefig(fig1_save)
plt.show()


fig2_name = 'costs_' + str(xGrid) + '.png'
fig2_save = os.path.join(figPath, fig2_name)

plt.figure()
plt.plot(itrPlot, costList, color = 'green')
plt.savefig(fig2_save)
plt.show()