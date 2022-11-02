# discretization points method version 1
# u(x, y) = 10x(1-x)y(1-y)(1-2y)
# u' w.r.t. x = 10(1-2x)y(1-y)(1-2y); u' w.r.t. y = 10x(1-x)(6y^2-6y+1)
# implement autograd gradient descent 
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
        # the error sum of NN derivative 
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
    # cost for the trained NN
    cost = 0.5 * (cost1 + cost2)
    return cost

def difference_NN_U(W, x, y):
    nnOut = NN(W, x, y)[0][0]
    fOut = U(x, y)
    diff = nnOut - fOut
    return diff

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

# evaluate whether to terminate training 
def terminate_evaluation(relErr, idx, itr, tol):
    # stop: 0, do not stop training; stop: 1, stop training
    stop = 0
    errlist = np.zeros(5)
    # idx out of iteration boundary, keep training
    if (idx + 5 > itr - 1):
        return stop
    # idx inside boundary 
    else:
        # take five of them 
        for i in range(0, 5):
            errlist[i] = relErr[idx + i]
        # the later one minus the current one
        for i in range(0, 4):
            # if the difference is larger than tol, keep training
            if (np.abs(errlist[i] - errlist[i + 1]) > tol):
                stop = 0
                return stop
            # if all four differences are less than tol, stop training
            else:
                stop = 1
        return stop

def adam(theta, grad, m, v, t, alpha, beta1, beta2, eps=1e-8):
    # update biased first moment estimate
    m = beta1 * m + (1.0 - beta1) * grad
    # update biased second raw moment estimate
    v = beta2 * v + (1.0 - beta2) * grad**2
    # compute bias-corrected first moment estimate
    mhat = m / (1.0 - beta1**(t + 1))
    # compute bias-corrected second raw moment estimate
    vhat = v / (1.0 - beta2**(t + 1))
    # update parameters
    theta = theta - alpha * mhat / (np.sqrt(vhat) + eps)
    return theta, m, v

# train neural network
def trainNN(lmb, itr, tol, H, W, xGrid, yGrid, eval_xGrid, eval_yGrid):
    # S: list of points on the boundry; T: list of points within the boundary
    S, T = generateCoordinates(xGrid, yGrid)
    # chosen points coordinate list
    s_list, t_list = generateCoordinates(eval_xGrid, eval_yGrid)
    points = s_list + t_list
    # count iteration times
    idx = 0

    # initialize Adam's parameters
    # factor for average gradient
    beta1 = 0.9
    # factor for average squared gradient 
    beta2 = 0.999
    # first moment
    m_weights_x = np.zeros(W[0][0].shape)
    m_weights_y = np.zeros(W[0][1].shape)
    m_actions = np.zeros(W[1].shape)
    m_bias = np.zeros(W[2].shape)
    # second moment vector
    v_weights_x = np.zeros(W[0][0].shape)
    v_weights_y = np.zeros(W[0][1].shape)
    v_actions = np.zeros(W[1].shape)
    v_bias = np.zeros(W[2].shape)

    # record cost list
    costList = np.zeros(itr)
    costList[idx] = costFunction(W, T, S)
    # record relative error for Sobolev Norm at each iteration
    relErr = np.zeros(itr)
    relErr[idx] = errorEval(W, points)

    # start counting wall clock time for traning NN
    startWall = time.time()
    # start counting CPU clock time for traning NN
    startCPU = time.process_time()

    # if the relative error differences are less than tolerence, stop training
    stop = terminate_evaluation(relErr, idx, itr, tol)

    # calculate partial derivative and update W
    while (stop == 0 and idx < itr - 1):
        costGrad = grad(costFunction, 0)(W, T, S)
        # apply adam stocasticate gradient descent to update weights and biases
        weight_x, m_weights_x, v_weights_x = adam(W[0][0], costGrad[0][0], m_weights_x, v_weights_x, idx, lmb, beta1, beta2, eps=1e-8)
        weight_y, m_weights_y, v_weights_y = adam(W[0][1], costGrad[0][1], m_weights_y, v_weights_y, idx, lmb, beta1, beta2, eps=1e-8)
        actions, m_actions, v_actions = adam(W[1], costGrad[1], m_actions, v_actions, idx, lmb, beta1, beta2, eps=1e-8)
        bias, m_bias, v_bias = adam(W[2], costGrad[2], m_bias, v_bias, idx, lmb, beta1, beta2, eps=1e-8)
        # update weights and biases
        W[0][0] = weight_x
        W[0][1] = weight_y
        W[1] = actions
        W[2] = bias
        # renew and record values
        idx += 1
        costList[idx] = costFunction(W, T, S)
        relErr[idx] = errorEval(W, points)
        # if the relative error differences are less than tolerence, stop training
        stop = terminate_evaluation(relErr, idx, itr, tol)
        # print training process
        print("iteration", idx)
        print("relErr", relErr[idx])
        print("cost", costList[idx])
        print("stop", stop)

    # end counting CPU clock time for training NN
    endCPU = time.process_time()
    cpuTime = endCPU - startCPU
    # end counting wall clock time for traning NN
    endWall = time.time()
    wallTime = endWall - startWall

    return W, cpuTime, wallTime, relErr, idx, costList

def record_results(W, cpuTime, wallTime, relErr, itrTimes, costList, H, M, K):
    # record results in text files
    current_path = os.path.abspath(__file__)
    filePath = os.path.join(os.path.abspath(os.path.dirname(current_path) + os.path.sep + "."),'records')
    # filePath = 'C:/Users/DELL/Desktop/DT/codes/records'
    now = time.strftime("%m%d%H%M")
    fileName = 'records_' + str(H) + '_' + now + '.txt'
    fileDir = os.path.join(filePath, fileName)

    file = open(fileDir, 'w')

    file.write('The number of points within the boundary M:' + str(M) +
                '\nThe number of points on the boundary K:' + str(K) +
                '\nThe number of grid points in x axis: ' + str(xGrid) +
                '\nThe number of grid points in y axis: ' + str(yGrid) +
                '\nThe number of hidden nodes: ' + str(H) +
                '\nLearning rate: ' + str(lmb) +
                '\nSetting Iteration number: ' + str(itr) +
                '\nMinimum relative errors for this H' + str(min(relErr)) + 
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

    fig1_name = 'relative_error_adam_' + str(H) + '.png'
    fig1_save = os.path.join(figPath, fig1_name)

    plt.figure()
    plt.plot(itrPlot, relErr, color = 'blue')
    plt.savefig(fig1_save)
    # plt.show()

    fig2_name = 'costs_adam_' + str(H) + '.png'
    fig2_save = os.path.join(figPath, fig2_name)

    plt.figure()
    plt.plot(itrPlot, costList, color = 'green')
    plt.savefig(fig2_save)
    # plt.show()
    return 0

# evaluate how the number of hidden nodes affect relative errors
def evaluate_H_relerror(lmb, itr, tol, xGrid, yGrid, eval_xGrid, eval_yGrid, M, K):
    # the number of hidden nodes needs to be evaluated
    H_num = 5
    # hidden nodes list 
    H_list = np.linspace(5, 25, H_num, dtype=int)
    # list containing minimum relative error for each H
    relerr_min_list = np.zeros(H_num)
    # H index
    i = 0
    for H in H_list:
        # initialize weight and bias: w: W[0], v: W[1], b: W[2]
        W = [npr.randn(2, H), npr.randn(H, 1), npr.randn(1, H)]
        # train a neural network w.r.t. parameter X and A
        W, cpuTime, wallTime, relErr, itrTimes, costList = trainNN(lmb, itr, tol, H, W, xGrid, yGrid, eval_xGrid, eval_yGrid)
        # record results in a file
        record_results(W, cpuTime, wallTime, relErr, itrTimes, costList, H, M, K)
        # append the minimum relative errors 
        relerr_min_list[i] = min(relErr)
        # print records on the command
        print("the number of hidden nodes", H)
        print("weight from input unit to hidden unit:", W[0])
        print("weight from hidden unit to the output:", W[1])
        print("bias:", W[2])
        print("wall time for training NN:", wallTime)
        print("CPU time for training NN:", cpuTime)
        print("iteration count:", itrTimes + 1)
        print("cost value at the final iteration:", costList[itrTimes])
        print("relative error for max norm at the final iteration:", relErr[itrTimes])
        print("the minimum relative errors for this H:", relerr_min_list[i])
        i += 1 
    return H_list, relerr_min_list


# number of points within the boundary
M = 2304 #3364
# number of points on the boundary
K = 196 #1196
#  number of grid points in x axis
xGrid = 50 # 300
#  number of grid points in y axis
yGrid = 50 # 300
# number of evaluation grid point in x axis
eval_xGrid = 60
# number of evaluation grid point in x axis
eval_yGrid = 60
# leaning rate lambda
lmb = 0.001
# iteratio n setting
itr = 3000
# tolerence rate setting
tol = 0.001

# investigate how the number of hidden nodes affects relative errors
H_list, relerr_min_list = evaluate_H_relerror(lmb, itr, tol, xGrid, yGrid, eval_xGrid, eval_yGrid, M, K)

# record results in text files
current_path = os.path.abspath(__file__)
filePath = os.path.join(os.path.abspath(os.path.dirname(current_path) + os.path.sep + "."),'records')
# filePath = 'C:/Users/DELL/Desktop/DT/codes/records'
now = time.strftime("%m%d%H%M")
fileName = 'records_H_relErr_' + now + '.txt'
fileDir = os.path.join(filePath, fileName)

file = open(fileDir, 'w')
file.write('hidden nodes list' + str(H_list) + 
           '\n minimum relative errors for each H' + str(relerr_min_list) +
           '\n number of dicretization points' + str(xGrid))
file.close()