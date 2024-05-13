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
        v_min=1e-2,v_max=50,
        input_min=-0.5,input_max=0.1,
        plot_derivative=False,
        flip_n=False,include_feedback_plots=True,
        colors=[colors[9-i] for i in range(10)]):
    
    if t_0 == None:
        t_0 = states.index[0]
    if n_0 == None:
        n_0 = np.sort(find_edge_location(pre_nonlinearity_states.loc[t_0],net.Npopulation))[0]
    
    if state_min == None:
        state_min = -net.J - 0.5
    if state_max == None:
        state_max = net.J + 0.5
    
    if include_feedback_plots:
        num_plots = 4
        plt.figure(figsize=(6,9))
    else:
        num_plots = 2
        plt.figure(figsize=(6,6))
    
    minInput,maxInput = 0,4
    minNeuron = n_0-num_neurons_scale
    maxNeuron = n_0+num_neurons_scale
    #times = [t_0,t_0*2,t_0*4]
    times = [t_0,t_0+25,t_0+50,t_0+75]
    
    # firing rate plot, edge neurons
    plt.subplot(num_plots,1,1)
    for time_index,t in enumerate(times):
        plt.plot(states.loc[t]['Neuron 0':'Neuron {}'.format(net.Npopulation-1)],
                 '.-',
                 label="$t$ = {}".format(int(t)),
                 lw=1,
                 ms=3,
                 color=colors[time_index])
    plt.ylabel('Neural state $x(n)$')
    nice_neuron_xlabels(net.Npopulation,
                        [minNeuron,n_0,maxNeuron])
    leg = plt.legend(framealpha=1)
    defaultFigure.makePretty(leg=leg)
    if flip_n:
        plt.axis(xmin=maxNeuron,xmax=minNeuron,
            ymin=state_min,ymax=state_max)
    else:
        plt.axis(xmin=minNeuron,xmax=maxNeuron,
            ymin=state_min,ymax=state_max)
    
    # firing rate plot, bump neurons
    plt.subplot(num_plots,1,2)
    for time_index,t in enumerate(times):
        plt.plot(
            states.loc[t]['Neuron {}'.format(net.Npopulation):'Neuron {}'.format(2*net.Npopulation-1)],
                 '.-',
                 label="$t$ = {}".format(t),
                 lw=1,
                 ms=3,
                 color=colors[time_index])
        if plot_derivative:
            # 2024/3/1 compare to actual discrete derivative of edge
            edge_states = np.array(states.loc[t]['Neuron 0':'Neuron {}'.format(net.Npopulation-1)])
            plt.plot(range(net.Npopulation-1),
                     edge_states[1:] - edge_states[:-1],
                     'k-',zorder=-10)
    plt.ylabel('Neural state $y(n)$')
    nice_neuron_xlabels(net.Npopulation,
                        [minNeuron,n_0,maxNeuron])
    #leg = plt.legend(framealpha=1)
    defaultFigure.makePretty()
    if flip_n:
        plt.axis(xmin=maxNeuron,xmax=minNeuron)
    else:
        plt.axis(xmin=minNeuron,xmax=maxNeuron)
    if bump_state_min != None:
        plt.axis(ymin=bump_state_min)
    if bump_state_max != None:
        plt.axis(ymax=bump_state_max)
    
    if include_feedback_plots:
    
        # interaction strength from bump to edge neurons
        plt.subplot(num_plots,1,3)
        plt.plot(abs(np.array(np.diag(net.bump_edge_Jmat))),
            color=colors[1])
        plt.ylabel('Feedback\nstrength $v(n)$')
        plt.yscale('log')
        nice_neuron_xlabels(net.Npopulation,
                            [minNeuron,n_0,maxNeuron])
        if flip_n:
            plt.axis(xmin=maxNeuron,xmax=minNeuron)
        else:
            plt.axis(xmin=minNeuron,xmax=maxNeuron)
        plt.axis(ymin=v_min,ymax=v_max)
        
        # input from bump neurons to edge neurons
        plt.subplot(num_plots,1,4)
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
                     color=colors[time_index])
        plt.ylabel("Feedback\n$\sum w_f g(y)$")
        nice_neuron_xlabels(net.Npopulation,
                            [minNeuron,n_0,maxNeuron])
        if flip_n:
            plt.axis(xmin=maxNeuron,xmax=minNeuron)
        else:
            plt.axis(xmin=minNeuron,xmax=maxNeuron)
        plt.axis(ymin=input_min,ymax=input_max)
    
    plt.xlabel('Neural unit, $n$')
    
    plt.subplots_adjust(bottom=0.1,top=0.95,left=0.2,right=0.95)

def edge_location_plot(net,states,n_0,t_0,delta_z,skip=10,logscale=False):
    plt.figure(figsize=(4,3))
    
    # plot edge location versus time
    plt.plot(states.index[::skip],[np.sort(find_edge_location(states.loc[i],net.Npopulation))[0] for i in states.index[::skip]],'-',label='Simulation',
        color='darkorange')
    
    # plot desired edge location versus time from theory
    theory_data = n_0+1./delta_z*np.log(states.index/t_0)
    theory_str = '$n_0+ (\Delta z)^{-1}\log(t/t_0)$'
    plt.plot(states.index,
             theory_data,
             'k',
             label=theory_str,
             zorder=-10,
             lw=1,
             ls=(0,(1,1)) )
             
    # add labels and adjust plot
    plt.xlabel('Time')
    plt.ylabel('Edge location, $\\bar n$')
    leg = plt.legend()
    defaultFigure.makePretty(leg=leg)
    #plt.savefig('231117_self_sustained_edge_location_vs_time.pdf')
    plt.axis(ymin=n_0-2)
    if logscale:
        plt.xscale('log')
    plt.subplots_adjust(left=0.21,right=0.95,bottom=0.2,top=0.95)
    
def time_rescaling_plot(states,n_0,t_0,delta_z,x_max,
        t_max_rescaled=8,
        state_min=-2.5,state_max=+2.5,
        delta_n=1,num_n_to_plot=10,
        neuron_indices=None,
        sim_type='past'):
    """
    x_max               : used to convert neural states to F(s)
    delta_n (1)         : indices of plotted neurons increment
                          by delta_n (ignored if
                          neuron_indices is specified)
                          (sign is flipped if sim_type='future')
    num_n_to_plot (10)  : number of neurons to plot (ignored if
                          neuron_indices is specified)
    """
    
    if sim_type=='future':
        delta_n = -delta_n
        t_max_rescaled = -t_max_rescaled
        
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
    plt.ylabel('Neural state $x$')
    plt.axis(ymin=state_min,ymax=state_max)
    plt.subplots_adjust(left=0.15,right=0.95)
    defaultFigure.makePretty(leg=leg)
    
    # plot rate over time for particular neurons, rescaled in time
    ax = plt.subplot(1,2,2)
    for i,neuron_index in enumerate(neuron_indices):
        name = 'Neuron {}'.format(neuron_index)
        times = states[name].index
        tau = abs(t_0)*np.exp((neuron_index-n_0)*delta_z)
        plt.plot(times/tau,states[name],label=name,
                 color=colors[i]) #str((neuron_index-n_0)/10))
    
    # plot inset using log axis
    if sim_type == 'future': # inset on upper left
        ax_inset = ax.inset_axes([0.15,0.45,0.5,0.5])
    elif sim_type == 'past': # inset on upper right
        ax_inset = ax.inset_axes([0.45,0.45,0.5,0.5])
    for i,neuron_index in enumerate(neuron_indices):
        name = 'Neuron {}'.format(neuron_index)
        ax_inset.plot(0.5*(states[name]/x_max + 1),label=name,
                 color=colors[i],lw=0.75)
        ax_inset.set_yscale('log')
    # no tick labels on inset
    ax_inset.set_xticklabels([])
    ax_inset.set_yticklabels([])
    ax_inset.axis(ymin=1e-2)
    ax_inset.set_xlabel('Time')
    ax_inset.set_ylabel('F')
    ax_inset.spines[['right', 'top']].set_visible(False)
    defaultFigure.makePretty(ax=ax_inset)
    
    #leg = plt.legend()
    plt.xlabel('Time/$\\tau_i$')
    plt.ylabel('Neural state $x$')
    plt.axis(ymin=state_min,ymax=state_max,
             xmin=min(0,t_max_rescaled),
             xmax=max(0,t_max_rescaled))
    plt.subplots_adjust(left=0.15,right=0.95)
    defaultFigure.makePretty(leg=leg)
    #plt.yscale('log')
    
    plt.subplots_adjust(wspace=0.3,bottom=0.2,top=0.95,
        left=0.1,right=0.825)
    
