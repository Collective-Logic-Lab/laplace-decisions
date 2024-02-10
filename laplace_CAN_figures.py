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

def nice_neuron_xlabels(Npopulation):
    labels=['' for i in range(Npopulation)]
    labeled_n = [0,int(Npopulation/2),Npopulation-1]
    for n in labeled_n:
        labels[n] = n
    plt.xticks(ticks=range(Npopulation),labels=labels)
    defaultFigure.makePretty()

def translation_simulation_plot(net,states,
        pre_nonlinearity_states,n_0,t_0,
        state_min=-2.5,state_max=2.5,
        input_min=-0.5,input_max=0.1):
    plt.figure(figsize=(6,9))
    minInput,maxInput = 0,4
    minNeuron,maxNeuron = n_0-30, n_0+30
    times = [t_0,t_0*2,t_0*4]
    
    # firing rate plot, edge neurons
    plt.subplot(4,1,1)
    for t in times:
        plt.plot(states.loc[t]['Neuron 0':'Neuron {}'.format(net.Npopulation-1)],
                 '.-',label="$t$ = {}".format(int(t)),lw=1,ms=3)
    #plt.hlines(0,0,50,color='k',lw=0.5)
    #plt.xlabel('Neural unit')
    plt.ylabel('State,\nedge neurons')
    nice_neuron_xlabels(net.Npopulation)
    leg = plt.legend(framealpha=1)
    defaultFigure.makePretty(leg=leg)
    plt.axis(xmin=minNeuron,xmax=maxNeuron,
        ymin=state_min,ymax=state_max)
    
    # firing rate plot, bump neurons
    plt.subplot(4,1,2)
    for t in times:
        plt.plot(
            states.loc[t]['Neuron {}'.format(net.Npopulation):'Neuron {}'.format(2*net.Npopulation-1)],
                 '.-',label="$t$ = {}".format(t),lw=1,ms=3)
    #plt.hlines(0,0,50,color='k',lw=0.5)
    #plt.xlabel('Neural unit')
    plt.ylabel('State,\nbump neurons')
    nice_neuron_xlabels(net.Npopulation)
    #leg = plt.legend(framealpha=1)
    defaultFigure.makePretty()
    plt.axis(xmin=minNeuron,xmax=maxNeuron,
             ymin=state_min,ymax=state_max)
    
    # interaction strength from bump to edge neurons
    plt.subplot(4,1,3)
    plt.plot(-np.array(np.diag(net.bump_edge_Jmat)))
    plt.ylabel('synaptic strength\nbump -> edge')
    #plt.xlabel('Neural unit')
    plt.yscale('log')
    nice_neuron_xlabels(net.Npopulation)
    plt.axis(xmin=minNeuron,xmax=maxNeuron,)
    
    # input from bump neurons to edge neurons
    plt.subplot(4,1,4)
    for t in times:
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
                 label="t = {}".format(t),lw=1,ms=3)
    #plt.hlines(0,0,50,color='k',lw=0.5)
    plt.xlabel('Neural unit')
    plt.ylabel('Input from bump\nto edge neurons')
    nice_neuron_xlabels(net.Npopulation)
    plt.axis(xmin=minNeuron,xmax=maxNeuron,
        ymin=input_min,ymax=input_max)
    
    plt.subplots_adjust(bottom=0.1,top=0.95,left=0.2,right=0.95)

def edge_location_plot(net,states,n_0,t_0,delta_z,skip=10):
    # plot edge location versus time
    plt.plot(states.index[::skip],[np.sort(abs(find_edge_location(states.loc[i])))[0] for i in states.index[::skip]],'.',label='Simulation')
    plt.plot(states.index,
             n_0+1./delta_z*np.log(states.index/t_0),
             label='$n_0+ (\Delta z)^{-1}\log(t/t_0)$',lw=2)
    plt.xlabel('Time $t$')
    plt.ylabel('Edge location (neuron number)')
    leg = plt.legend()
    defaultFigure.makePretty(leg=leg)
    #plt.savefig('231117_self_sustained_edge_location_vs_time.pdf')
    plt.axis(ymin=n_0-2)
    plt.xscale('log')
    
def time_rescaling_plot(states,n_0,t_0,delta_z,
        t_max=80,t_max_rescaled=8,
        state_min=-2.5,state_max=+2.5):
    plt.figure(figsize=(10,3))
    
    # plot state over time for particular neurons
    plt.subplot(1,2,1)
    neuron_indices = range(n_0,n_0+10)
    
    for neuron_index in neuron_indices:
        name = 'Neuron {}'.format(neuron_index)
        plt.plot(states[name],label=name,
                 color='C{}'.format(neuron_index-n_0)) #color=str((neuron_index-n_0)/10))
    leg = plt.legend(loc=(2.4,-0.03))
    plt.xlabel('Time')
    plt.ylabel('Neural state')
    plt.axis(ymin=state_min,ymax=state_max)
    plt.subplots_adjust(left=0.15,right=0.95)
    defaultFigure.makePretty(leg=leg)
    #plt.savefig('231018_firing_rate_vs_time.pdf')
    
    # plot rate over time for particular neurons, rescaled in time
    plt.subplot(1,2,2)
    neuron_indices = range(n_0,n_0+10)
    for neuron_index in neuron_indices:
        name = 'Neuron {}'.format(neuron_index)
        times = states[name].index
        tau = t_0*np.exp((neuron_index-n_0)*delta_z)
        plt.plot(times/tau,states[name],label=name,
                 color='C{}'.format(neuron_index-n_0)) #str((neuron_index-n_0)/10))
    
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
    