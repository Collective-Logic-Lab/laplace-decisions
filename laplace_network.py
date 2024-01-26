# laplace_network.py
#
# 2023/11/9
# Bryan Daniels
#
# A "continuous attractor"-style model that produces the
# Laplace transform picture of Howard et al. 2018 and is
# implemented using believable neural dynamics
#

import simpleNeuralModel
import numpy as np
import scipy.stats

def gaussian_kernel_matrix(N,width,normed=True):
    """
    Interaction matrix with Gaussian kernel.
    
    N:       total number of units.  Returned interaction matrix will have shape (N,N)
    width:   width of the Gaussian measured in number of units
    """
    mat = [ scipy.stats.distributions.norm.pdf(range(N),loc=i,scale=width) for i in range(N) ]
    if normed:
        mat = [ row/np.sum(row) for row in mat ]
    return np.array(mat)

def derivative_interaction_matrix(N):
    """
    Interaction matrix J_ij designed to give inputs proportional to the spatial derivative
    of the column (j) neurons in the row (i) neurons.
    (Row neurons at each end receive no input.)
    
    N:      Number of units in each population.
            Returned interaction matrix will have shape (N,N).
    """
    mat = np.diag(-np.ones(N-1),k=-1) + np.diag(np.ones(N-1),k=1)
    # row neurons at each end receive no input
    mat[0] = np.zeros(N)
    mat[N-1] = np.zeros(N)
    return mat
    
def interaction_matrix_from_kernel(discreteKernel,N,normed=True):
    """
    Takes a 1D list specifying the shape of an interaction kernel and produces an
    interaction matrix for neurons lying along a 1D space.
    
    The kernel is assumed to be zero outside of the range [-N/2,N/2].
    
    discreteKernel       : Should have an odd length <= N.  The kernel is centered on the
                           middle element (index (len(discreteKernel)-1)/2).
    N                    : The number of neurons (dimension of the output matrix is NxN)
    normed (True)        : If True, each row is normalized to sum to 1.
    """
    Nk = len(discreteKernel)
    assert(Nk%2==1) # discreteKernel should have odd length
    assert(Nk <= N) # discreteKernel should have length less than or equal to N
    mat = np.zeros((N,N))
    
    # copy in the appropriate part of the kernel for each row
    for i in range(N):
        matIndexMin = max(0,i-(Nk-1)//2)
        matIndexMax = min(i+Nk-(Nk-1)//2,N)
        kIndexMin = 0 + max(0,matIndexMin - (i-(Nk-1)//2))
        kIndexMax = Nk - max(0,i+Nk-(Nk-1)//2 - N)
        mat[i,matIndexMin:matIndexMax] = discreteKernel[kIndexMin:kIndexMax]
        
    if normed:
        mat = [ row/np.sum(row) for row in mat ]
    return np.array(mat)

def find_edge_location(rates_series,k=1):
    """
    Takes a pandas Series (or simple list) of rates or states along the 1D line of neurons.
    Returns the (interpolated) location of the zero crossing.
    """
    ppoly = interpolated_state(rates_series,k=k)
    return ppoly.roots()

def interpolated_state(rates_series,k=1):
    """
    Takes a pandas Series (or simple list) of rates or states along the 1D line of neurons.
    Returns the scipy.interpolate.PPoly function object representing an interpolated spline.
    """
    tck = scipy.interpolate.splrep(range(len(rates_series)),rates_series,k=k)
    ppoly = scipy.interpolate.PPoly.from_spline(tck)
    return ppoly

class laplace_network:
    
    def __init__(self,Npopulation,J=1,kernel_width=2,boundary_input=100,
        num_inputs=5,include_bump=True,J_edge_bump=1,J_bump_edge=1,
        nonlinearity=np.tanh,sigma=1):
        """
        Create 1-D line of units with nearest-neighbor interactions and fixed
        boundary conditions implemented by large fields at the ends.
        
        Npopulation    : number of units per population
                         (if including bump neurons, total number of neurons is 2N)
        J              : scale of interaction strength among nearby neighbors
        kernel_width   : width of Gaussian kernel for interactions
        boundary_input : field setting boundary conditions (negative on left end
                         and positive on right end)
        num_inputs     : number of fixed input nodes at each end of the edge neurons
        include_bump   : If True, include N additional neurons that encode the derivative
                         of the edge neurons.
        J_edge_bump    : scale of interaction strength of edge -> bump connections
        J_bump_edge    : scale of interaction strength of bump -> edge connections
                         (can be a scalar or a vector of length Npopulation)
        nonlinearity   : Function taking neural states to synaptic currents.
                         Default is np.tanh.  See simpleNeuralModel.
        sigma          : Scale of nonlinearity function.  See simpleNeuralModel.
        """
        self.Npopulation = Npopulation
        self.J = J
        self.kernel_width = kernel_width
        self.boundary_input = boundary_input
        self.num_inputs = num_inputs
        self.include_bump = include_bump
        self.nonlinearity = nonlinearity
        self.sigma = sigma
        
        # set interaction matrix for edge neurons -> edge neurons
        self.edge_Jmat = J * gaussian_kernel_matrix(Npopulation,kernel_width)
        
        if include_bump:
            # set interaction matrix for bump neurons -> bump neurons
            self.bump_Jmat = np.zeros((Npopulation,Npopulation))
            
            # set interaction matrix for edge neurons -> bump neurons
            self.edge_bump_Jmat = J_edge_bump * derivative_interaction_matrix(Npopulation)
            
            # set interaction matrix for bump neurons -> edge neurons
            self.bump_edge_Jmat = np.diag(J_bump_edge * np.ones(Npopulation))
        
            # construct full interaction matrix
            self.Jmat = np.block([[self.edge_Jmat, self.bump_edge_Jmat],
                                  [self.edge_bump_Jmat, self.bump_Jmat]])
                                  
            # also store interaction matrix that does not include feedback from bump to edge
            self.Jmat_no_feedback = np.block([[self.edge_Jmat, np.zeros((Npopulation,Npopulation))],
                                              [self.edge_bump_Jmat, self.bump_Jmat]])
        else:
            self.Jmat = self.edge_Jmat
            self.Jmat_no_feedback = self.edge_Jmat
        
        self.Ntotal = len(self.Jmat)
        
        # set external inputs to edge neurons
        inputExt = np.zeros(self.Ntotal)
        inputExt[0:num_inputs] = -boundary_input
        inputExt[self.Npopulation-num_inputs:self.Npopulation] = boundary_input
        self.inputExt = inputExt
        
    def find_edge_state(self,center,initial_guess_edge=None,method='translate'):
        """
        Find stationary state (fixed point) that looks like an edge at the given location
        within the "edge" neurons.
        
        If the network includes "bump" neurons, feedback from the bump neurons to the edge
        neurons is neglected here.

        center                    : desired center location of edge
        initial_guess_edge (None) : Optionally give an initial guess for the state of edge
                                    neurons in the edge state.  If None, a default is used.
                                    (The default is designed to work with the standard Gaussian
                                    interaction kernel.)
        method ('translate')      : If 'translate', first find the edge fixed point numerically
                                    in the middle of the network, then interpolate and translate
                                    to the desired position.  Can be more numerically stable when
                                    taking derivatives.
                                    If 'minimize', find the edge fixed point numerically directly
                                    at the desired position.
        
        """
        if method=='minimize':
            initial_location = center
        elif method=='translate':
            initial_location = self.Npopulation/2
        else:
            raise Exception('Unrecognized method: {}'.format(method))
        
        # set initial guess state
        if initial_guess_edge is None:
            # TO DO: should the edge width be equal to the kernel width? (seems to work...)
            width = self.kernel_width
            initial_guess_edge = (np.arange(0,self.Npopulation)-initial_location)/width
        if self.include_bump:
            initialGuessState = np.concatenate([initial_guess_edge,
                                                np.zeros(self.Npopulation)])
        else:
            initialGuessState = initial_guess_edge
        assert(np.shape(initialGuessState)==(self.Ntotal,))
            
        # find edge state numerically
        fp_initial = simpleNeuralModel.findFixedPoint(self.Jmat_no_feedback,
                                                      initialGuessState,
                                                      inputExt=self.inputExt,
                                                      nonlinearity=self.nonlinearity,
                                                      sigma=self.sigma)
        
        # if requested, move the edge to the desired location
        if method=='translate':
            # start by keeping the states of the end inputs plus padding of 2*kernel_width fixed,
            # and with saturated left and right states everywhere else around the desired center
            fp = fp_initial.copy()
            fixed_end_width = self.num_inputs + int(np.ceil(2*self.kernel_width))
            left_state = fp_initial[fixed_end_width]
            right_state = fp_initial[self.Npopulation-fixed_end_width-1]
            fp[fixed_end_width:int(center)] = left_state
            fp[int(center):self.Npopulation-fixed_end_width] = right_state
            
            # now interpolate the states around the initial edge and paste this in the new location
            initial_actual_location = find_edge_location(fp_initial)[0]
            shift = center - initial_actual_location
            fp_initial_spline = interpolated_state(fp_initial)
            # set up range of locations that will be overwritten
            n_min = max(fixed_end_width,
                        fixed_end_width+int(np.ceil(shift)))
            n_max = min(self.Npopulation-fixed_end_width,
                        self.Npopulation-fixed_end_width+int(np.floor(shift)))
            n_vals = range(n_min,n_max)
            # overwrite with the shifted edge
            fp[n_vals] = fp_initial_spline(n_vals-shift)
            
            if self.include_bump:
                # also shift states of bump neurons in a similar way
                
                # start by keeping the states of the end inputs plus padding
                # of 2*kernel_width fixed, and with zeros everywhere in between
                n_min_middle_bump = self.Npopulation+fixed_end_width
                n_max_middle_bump = 2*self.Npopulation-fixed_end_width
                n_vals_middle_bump = range(n_min_middle_bump,n_max_middle_bump)
                fp[n_vals_middle_bump] = np.zeros_like(n_vals_middle_bump)
                
                # now paste the interpolated bump in the new location
                n_min_bump = max(self.Npopulation+fixed_end_width,
                                 self.Npopulation+fixed_end_width+int(np.ceil(shift)))
                n_max_bump = min(2*self.Npopulation-fixed_end_width,
                                 2*self.Npopulation-fixed_end_width+int(np.floor(shift)))
                n_vals_bump = range(n_min_bump,n_max_bump)
                # overwrite with the shifted bump
                fp[n_vals_bump] = fp_initial_spline(n_vals_bump-shift)
        else:
            fp = fp_initial
        
        return fp
    
    def simulate_dynamics(self,initial_state,t_final,noise_var,
        additional_input=None,seed=None,delta_t=0.001):
        """
        Use simpleNeuralModel.simpleNeuralDynamics to simulate the network's dynamics.

        additional_input (None)      : If given a list of length Ntotal, add this to the existing
                                         external current as a constant input.
                                       If given an array of shape (# timepoints)x(Ntotal), add this
                                         to the existing external current as an input that
                                         varies over time.  (# timepoints = t_final/delta_t)
        seed (None)                  : If given, set random seed before running
        """
        num_timepoints = t_final/delta_t
        if additional_input is not None:
            if np.shape(additional_input) == (self.Ntotal,):
                total_input = self.inputExt + additional_input
            elif np.shape(additional_input) == (num_timepoints,self.Ntotal):
                total_input = [ self.inputExt + a for a in additional_input ]
            else:
                raise Exception("Unrecognized form of additional_input")
        else:
            total_input = self.inputExt
        
        if seed is not None:
            np.random.seed(seed)

        return simpleNeuralModel.simpleNeuralDynamics(self.Jmat,
                                                      total_input,
                                                      noiseVar=noise_var,
                                                      tFinal=t_final,
                                                      initialState=initial_state,
                                                      deltat=delta_t,
                                                      nonlinearity=self.nonlinearity,
                                                      sigma=self.sigma)
