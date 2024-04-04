# laplace_CAN_figures.py
#
# Bryan Daniels
# 2024/2/9
#
# Plots to be used in laplace-CAN-paper-figures.ipynb.
#
# 2024/2/7 branched from develop-exponential-decay-transition.ipynb
# 2024/1/25
#

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from toolbox import defaultFigure
defaultFigure.setDefaultParams()
from laplace_network import find_edge_location

# colors from coolors.co
colors = {0: "#086788", # cerulean
          1: "#07A0C3", # blue green
          2: "#F0C808", # jonquil
          3: "#DD1C1A", # rojo
          4: "#BEB2C8", # thistle
          5: "#6B2737", # wine
          6: "#966B9D", # pomp and power
          7: "#FF6663", # bittersweet
          8: "#20BF55", # dark pastel green
          9: "#440381", # indigo
         }

def nice_neuron_xlabels(Npopulation,labeled_n=None):
    labels=['' for i in range(Npopulation)]
    if labeled_n == None:
        labeled_n = [0,
                     int(Npopulation/2),
                     Npopulation-1]
    for n in labeled_n:
        labels[int(n)] = int(n)
    plt.xticks(ticks=range(Npopulation),labels=labels)
    defaultFigure.makePretty()

def translation_simulation_plot(net,states,
        pre_nonlinearity_states,
        num_neurons_scale=30,
        n_0=None,t_0=None,
        state_min=None,state_max=None,
        bump_state_min=None,bump_state_max=None,
        input_min=-0.5,input_max=0.1,
        plot_derivative=False):
    
    if t_0 == None:
        t_0 = states.index[0]
    if n_0 == None:
        n_0 = np.sort(find_edge_location(pre_nonlinearity_states.loc[t_0],net.Npopulation))[0]
    
    if state_min == None:
        state_min = -net.J - 0.5
    if state_max == None:
        state_max = net.J + 0.5
    
    plt.figure(figsize=(6,9))
    minInput,maxInput = 0,4
    minNeuron = n_0-num_neurons_scale
    maxNeuron = n_0+num_neurons_scale
    times = [t_0,t_0*2,t_0*4]
    
    # firing rate plot, edge neurons
    plt.subplot(4,1,1)
    for time_index,t in enumerate(times):
        plt.plot(states.loc[t]['Neuron 0':'Neuron {}'.format(net.Npopulation-1)],
                 '.-',
                 label="$t$ = {}".format(int(t)),
                 lw=1,
                 ms=3,
                 color=colors[9-time_index])
    #plt.hlines(0,0,50,color='k',lw=0.5)
    #plt.xlabel('Neural unit')
    plt.ylabel('State,\nedge neurons')
    nice_neuron_xlabels(net.Npopulation,
                        [minNeuron,n_0,maxNeuron])
    leg = plt.legend(framealpha=1)
    defaultFigure.makePretty(leg=leg)
    plt.axis(xmin=minNeuron,xmax=maxNeuron,
        ymin=state_min,ymax=state_max)
    
    # firing rate plot, bump neurons
    plt.subplot(4,1,2)
    for time_index,t in enumerate(times):
        plt.plot(
            states.loc[t]['Neuron {}'.format(net.Npopulation):'Neuron {}'.format(2*net.Npopulation-1)],
                 '.-',
                 label="$t$ = {}".format(t),
                 lw=1,
                 ms=3,
                 color=colors[9-time_index])
        if plot_derivative:
            # 2024/3/1 compare to actual discrete derivative of edge
            edge_states = np.array(states.loc[t]['Neuron 0':'Neuron {}'.format(net.Npopulation-1)])
            plt.plot(range(net.Npopulation-1),
                     edge_states[1:] - edge_states[:-1],
                     'k-',zorder=-10)
    #plt.hlines(0,0,50,color='k',lw=0.5)
    #plt.xlabel('Neural unit')
    plt.ylabel('State,\nbump neurons')
    nice_neuron_xlabels(net.Npopulation,
                        [minNeuron,n_0,maxNeuron])
    #leg = plt.legend(framealpha=1)
    defaultFigure.makePretty()
    plt.axis(xmin=minNeuron,xmax=maxNeuron)
    if bump_state_min != None:
        plt.axis(ymin=bump_state_min)
    if bump_state_max != None:
        plt.axis(ymax=bump_state_max)
    
    # interaction strength from bump to edge neurons
    plt.subplot(4,1,3)
    plt.plot(abs(np.array(np.diag(net.bump_edge_Jmat))),
        color=colors[1])
    plt.ylabel('synaptic strength\nbump -> edge')
    #plt.xlabel('Neural unit')
    plt.yscale('log')
    nice_neuron_xlabels(net.Npopulation,
                        [minNeuron,n_0,maxNeuron])
    plt.axis(xmin=minNeuron,xmax=maxNeuron,)
    
    # input from bump neurons to edge neurons
    plt.subplot(4,1,4)
    for time_index,t in enumerate(times):
        if np.shape(net.sigma) == (net.Ntotal,net.Ntotal):
            # activities_t has shape (NtotalxNtotal)
            activities_t = np.tanh(np.tile(
                pre_nonlinearity_states.loc[t],
                (net.Ntotal,1))/net.sigma)
            # bump_to_edge_activities
            # has shape (NpopulationxNpopulation)
            bump_to_edge_activities = \
                activities_t[:net.Npopulation,
                            net.Npopulation:2*net.Npopulation]
            # bump_to_edge_input has shape (N)
            bump_to_edge_input = np.sum(
                net.bump_edge_Jmat*np.array(
                    bump_to_edge_activities),axis=1)
            bump_to_edge_input = pd.Series(bump_to_edge_input,
                index=pre_nonlinearity_states.columns[:net.Npopulation])
        else:
            activities = np.tanh(
                pre_nonlinearity_states/net.sigma)
            
            bumpActivities = activities.loc[t][
                'Neuron {}'.format(net.Npopulation):'Neuron {}'.format(2*net.Npopulation-1)]
            bump_to_edge_input = np.dot(bumpActivities,
                                        net.bump_edge_Jmat)
        plt.plot(bump_to_edge_input,'.-',
                 label="t = {}".format(t),
                 lw=1,
                 ms=3,
                 color=colors[9-time_index])
    #plt.hlines(0,0,50,color='k',lw=0.5)
    plt.xlabel('Neural unit')
    plt.ylabel('Input from bump\nto edge neurons')
    nice_neuron_xlabels(net.Npopulation,
                        [minNeuron,n_0,maxNeuron])
    plt.axis(xmin=minNeuron,xmax=maxNeuron,
        ymin=input_min,ymax=input_max)
    
    plt.subplots_adjust(bottom=0.1,top=0.95,left=0.2,right=0.95)

def edge_location_plot(net,states,n_0,t_0,delta_z,skip=10):
    plt.figure(figsize=(4,3))
    
    # plot edge location versus time
    plt.plot(states.index[::skip],[np.sort(find_edge_location(states.loc[i],net.Npopulation))[0] for i in states.index[::skip]],'.',label='Simulation',
        color='k')
    plt.plot(states.index,
             n_0+1./delta_z*np.log(states.index/t_0),
             label='$n_0+ (\Delta z)^{-1}\log(t/t_0)$',lw=2,
             color='crimson')
    plt.xlabel('Time')
    plt.ylabel('Edge location\n(neuron number)')
    leg = plt.legend()
    defaultFigure.makePretty(leg=leg)
    #plt.savefig('231117_self_sustained_edge_location_vs_time.pdf')
    plt.axis(ymin=n_0-2)
    plt.xscale('log')
    plt.subplots_adjust(left=0.21,right=0.95,bottom=0.2,top=0.95)
    
def time_rescaling_plot(states,n_0,t_0,delta_z,
        t_max=80,t_max_rescaled=8,
        state_min=-2.5,state_max=+2.5,
        delta_n=1,num_n_to_plot=10,
        neuron_indices=None):
    """
    delta_n (1)         : indices of plotted neurons increment
                          by delta_n (ignored if
                          neuron_indices is specified)
    num_n_to_plot (10)  : number of neurons to plot (ignored if
                          neuron_indices is specified)
    """
    
    if neuron_indices == None:
        neuron_indices = range(int(n_0),
                               int(n_0)+delta_n*num_n_to_plot,
                               delta_n)
    
    plt.figure(figsize=(10,3))
    
    # plot state over time for particular neurons
    plt.subplot(1,2,1)
    
    for i,neuron_index in enumerate(neuron_indices):
        name = 'Neuron {}'.format(neuron_index)
        plt.plot(states[name],label=name,
                 color=colors[i]) #color=str((neuron_index-n_0)/10))
    leg = plt.legend(loc=(2.4,-0.03))
    plt.xlabel('Time')
    plt.ylabel('Neural state')
    plt.axis(ymin=state_min,ymax=state_max)
    plt.subplots_adjust(left=0.15,right=0.95)
    defaultFigure.makePretty(leg=leg)
    #plt.savefig('231018_firing_rate_vs_time.pdf')
    
    # plot rate over time for particular neurons, rescaled in time
    plt.subplot(1,2,2)
    for i,neuron_index in enumerate(neuron_indices):
        name = 'Neuron {}'.format(neuron_index)
        times = states[name].index
        tau = t_0*np.exp((neuron_index-n_0)*delta_z)
        plt.plot(times/tau,states[name],label=name,
                 color=colors[i]) #str((neuron_index-n_0)/10))
    
    #leg = plt.legend()
    plt.xlabel('Time/$\\tau_i$')
    plt.ylabel('Neural state')
    plt.axis(ymin=state_min,ymax=state_max,
        xmin=0,xmax=t_max_rescaled)
    plt.subplots_adjust(left=0.15,right=0.95)
    defaultFigure.makePretty(leg=leg)
    #plt.yscale('log')
    
    plt.subplots_adjust(wspace=0.3,bottom=0.2,top=0.95,
        left=0.1,right=0.825)
    
