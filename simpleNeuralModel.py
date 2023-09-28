# simpleNeuralModel.py
#
# Bryan Daniels
# 2023/8/25 branched from CAS-503-Collectives/neural
# 2021/9/10
#
# A simple model of neural dynamics.  This is equivalent to the model found in the
# following reference:
#     Daniels, Bryan C., Jessica C. Flack, and David C. Krakauer.
#     “Dual Coding Theory Explains Biphasic Collective Computation in Neural
#     Decision-Making.” Frontiers in Neuroscience 11, 1–16 (2017).
#     https://doi.org/10.3389/fnins.2017.00313
#

import numpy as np
import scipy.optimize as opt
import pandas as pd

def simpleNeuralDynamics(weightMatrix,inputExt=0,noiseVar=1,
    tFinal=10,deltat=1e-3,initialState=None):
    """
    Simulates the following stochastic process:
    
    dx_i / dt = inputExt - x_i + sum_j weightMatrix_{i,j} tanh(x_j) + xi
    
    where xi is uncorrelated Gaussian noise with variance 'noiseVar' per unit time.
    
    Time is discretized into units of deltat, and the simulation is run until time tFinal.
    
    weightMatrix                      : (N x N) matrix indicating the synaptic strength from
                                        neuron j to neuron i
    inputExt (0)                      : If given a constant or list of length N, add this as
                                          a constant external input.
                                        If given an array of shape (# timepoints)x(N),
                                          add external current as an input that
                                          varies over time.  (# timepoints = t_final/delta_t)
    initialState (None)               : If given a list of length N, start the system in the
                                        given state.  If None, initial state defaults to
                                        all zeros.
    """
    N = len(weightMatrix)
    # make sure the weight matrix is square
    assert(len(weightMatrix[0])==N)
    
    # set up the initial state
    if initialState is None:
        initialState = np.zeros(N)
    # make sure the initial state has the correct length
    assert(len(initialState)==N)
    
    # set up the simulation times and a list to hold the simulated steps
    times = np.arange(0,tFinal+deltat,deltat)
    stateList = [initialState,]
    
    # set up the external input, possibly varying as a function of time
    if np.shape(inputExt) == (len(times)-1,N):
        # input is varying in time
        inputExtVsT = inputExt
    elif np.shape(inputExt) == () or np.shape(inputExt) == (N,):
        # input is constant in time
        inputExtVsT = [ inputExt for t in times[:-1] ]
    else:
        raise Exception("Unrecognized form of inputExt")
        
    
    # run the simulation (we already have the state for t=0)
    for time,inputCurrent in zip(times[1:],inputExtVsT):
        currentState = stateList[-1]
        
        # compute deltax for current timestep
        deterministicPart = deltat*( inputCurrent - currentState + np.dot(weightMatrix,np.tanh(currentState)) )
        stochasticPart = np.sqrt(deltat*noiseVar)*np.random.normal(size=N)
        deltax = deterministicPart + stochasticPart
        
        # update to find the new state
        newState = currentState + deltax
        
        # record the new state
        stateList.append(newState)
       
    # return simulation output as a pandas dataframe
    df = pd.DataFrame(stateList,index=times,columns=['Neuron {}'.format(i) for i in range(N)])
    df.index.set_names('Time',inplace=True)
    return df

def allToAllNetworkAdjacency(N):
    return 1 - np.eye(N)

def findFixedPoint(weightMatrix,initialGuessState,inputExt=0):
    """
    Find a fixed point of the deterministic part of dynamics
    """
    N = len(weightMatrix)
    # make sure the input is either a simple number or length-N
    assert(np.shape(inputExt)==() or np.shape(inputExt)==(N,))
    
    deterministicDeltaX = lambda x: inputExt - x + np.dot(weightMatrix,np.tanh(x))
    sol = opt.root(deterministicDeltaX,initialGuessState)
    return sol.x

def findFixedPoints(weightMatrix,inputExt=0,useMeanField=True,startMin=-10,
    startMax=10,numToTest=100):
    """
    look for all fixed points nearby a set of starting points
    """
    N = len(weightMatrix)
    # make sure the input is either a simple number or length-N
    assert(np.shape(inputExt)==() or np.shape(inputExt)==(N,))
    
    fixedPointList = []
    if useMeanField and np.mean(np.sum(weightMatrix,axis=0)) > 1.:
        xMF = 2.*np.sqrt(np.mean(np.sum(weightMatrix,axis=0))-1.)
        startingPoints = [-xMF,0.,+xMF]
    else:
        startingPoints = list(np.linspace(startMin,startMax,numToTest))
    for startingPoint in startingPoints:
        initialGuessState = startingPoint*np.ones(N)
        fixedPoint = findFixedPoint(weightMatrix,initialGuessState,inputExt=inputExt)
        fixedPointList.append(fixedPoint)
    uniqueFixedPoints = np.unique(np.round(fixedPointList,5),axis=0)
    return pd.DataFrame(uniqueFixedPoints,columns=['Neuron {}'.format(i) for i in range(N)])
