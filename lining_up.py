## Lining up by state transitions

import os
import tables
import numpy as np
import pylab as plt
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg

data_direc= '/media/sf_shared_folder/jian_you_data/'
os.chdir(data_direc)
hf5 = tables.open_file('jy05_20170324_2500mslaser_170324_103823_repacked.h5')
states = range(3,7);

plot_dir = './correlation_analysis_plots'
if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)

################
## OFF TRIALS ##
################
#states = hf5.list_nodes('/spike_trains/multinomial_hmm_results/laser_off')
#dig_in = [dig_in[i] if dig_in[i].__str__().find('dig_in')!=-1 else None for i in range(len(dig_in))]

for num_states in range(len(states)):
    ## Loop through states
    min_t = 700
    max_t = 1200
    exec('post_prob = hf5.root.spike_trains.multinomial_hmm_results.laser_off.states_%i.posterior_proba[:]' % states[num_states]) 
    #post_prob shape = [trials,time,states]
    exec('t = np.array(hf5.root.spike_trains.multinomial_hmm_results.laser_off.states_%i.time[:])' % states[num_states])
    indices = np.where((t >= min_t)*(t <= max_t))[0]
    
    ## Note : HMM model is fit to particular trials but predicts posterior
    ## probabilities for transitions in all trials.
    
    # list all tastes
    taste_n = 30
    dig_in = hf5.list_nodes('/spike_trains')
    dig_in = [dig_in[i] if dig_in[i].__str__().find('dig_in')!=-1 else None for i in range(len(dig_in))]
    dig_in = list(filter(None, dig_in))
    
    off_trials = np.concatenate(tuple(np.where(dig_in[i].laser_durations[:] == 0)[0] + taste_n*i for i in range(len(dig_in))))

    post_prob = post_prob[off_trials,:,:]
    ## Since we're indexing post_prob by off_trials, EVERYTHING has to be indexed by off_trials
    
    sum_state_mass = np.sum(post_prob[:, indices, :], axis = (0,1)) #to find index of state with highest occurence during set time
    #transition = np.where(post_prob[:,:,np.argmax(sum_state_mass)] >= 0.8) #where[0] = x = trial, where[1] = y = time
    max_state = np.argmax(sum_state_mass)
    
    transition = []
    for i in range(post_prob.shape[0]):
        this_transition = np.where(post_prob[i, :, max_state] >= 0.8)[0]
        if len(this_transition) > 0:
            transition.append([i, this_transition[0]])
    
    transition_trials = np.array([transition[i][0] for i in range(len(transition))])
    ## Now EVERYTHING has to be indexed by transition trials as well -- MAYBE NOT
    fin_off_trials = off_trials[transition_trials]
    
            
    palatability_order = [3,4,1,2]
    trial_palatability = np.zeros(len(palatability_order)*30)
    for i in range(0,len(palatability_order)):
        trial_palatability[i*30:(i+1)*30] = np.repeat(palatability_order[i],30)[:]
    #trial_palatability = np.array(trial_palatability)
    
    # Get the spike array for each individual taste and merge all:
    spikes = np.concatenate(tuple(dig_in[i].spike_array[:] for i in range(len(dig_in))), axis = 0)
    off_spikes = spikes[off_trials,:,:]
    
    # Take spikes from trials with transitions and align them according to state transition per trial
    pre_stim_t = 2000   #time before stimulus
    data_window_pre = 500
    data_window_post = 750
    lined_spikes = []
    for trial, time in transition:
        lined_spikes.append(off_spikes[trial, :, pre_stim_t + t[time] - data_window_pre :pre_stim_t + t[time] + data_window_post])
    # lined_spikes shape = list of trials, every index = array[neurons , time]
    
    #for trial in transition[0]:
    #    for time in transition[1]:
    #        lined_spikes.append(spikes[trial, :, 1500 + t[time]:2500 + t[time]])
            
    lined_spikes = np.array(lined_spikes) # [trials, neurons, time]
    
    bin_window_size = 250
    step_size = 25
    ## Making PSTHs for the spikes
    lined_firing = np.zeros((lined_spikes.shape[0], lined_spikes.shape[1], int((data_window_pre + data_window_post - bin_window_size)/step_size) + 1))
    
    for i in range(lined_firing.shape[0]):
        for j in range(lined_firing.shape[1]):
            for k in range(lined_firing.shape[2]):
                lined_firing[i, j, k] = np.mean(lined_spikes[i, j, step_size*k:step_size*k + bin_window_size])
                
    test = lined_firing
    test = np.sum(test,0)
    for i in range(test.shape[0]):
        test[i,:] = test[i,:]/test[i,0]
    
    fig = plt.figure()
    plt.title('Aligned firing rates, Off trials , Model states =' + str(states[num_states]))
    for i in range(test.shape[0]):
        plt.plot(test[i,:])
    fig.savefig(plot_dir + '/off_firing_states%i.png' % states[num_states])
    plt.close('all')
                
    #palatability = np.ones(lined_firing.shape[0])
    #palatability = np.array([trial_palatability[transition[i][0]] for i in range(0,len(transition))])
    palatability = trial_palatability[fin_off_trials]
    
    r_pearson = np.zeros((lined_firing.shape[1], lined_firing.shape[2]))
    p_pearson = np.ones((lined_firing.shape[1], lined_firing.shape[2]))
    
    for i in range(r_pearson.shape[0]):
        for j in range(r_pearson.shape[1]):
            r_pearson[i, j], p_pearson[i, j] = pearsonr(lined_firing[:, i, j], palatability)
            if np.isnan(r_pearson[i, j]):
                r_pearson[i, j] = 0.0
                p_pearson[i, j] = 1.0
    
    fig = plt.figure()  
    plt.title('Correlation, Off trials , Model states =' + str(states[num_states]))
    plt.plot(np.mean(r_pearson[:, :]**2, axis = 0))
    fig.savefig(plot_dir + '/off_correlation_states%i.png' % states[num_states])
    plt.close('all')

###############
## ON TRIALS ##
###############
    
for num_states in range(len(states)):
    ## Loop through states
    min_t = 700
    max_t = 1200
    exec('post_prob = hf5.root.spike_trains.multinomial_hmm_results.laser_on.states_%i.posterior_proba[:]' % states[num_states]) #shape = [trials,time,states]
    exec('t = np.array(hf5.root.spike_trains.multinomial_hmm_results.laser_on.states_%i.time[:])' % states[num_states])
    indices = np.where((t >= min_t)*(t <= max_t))[0]
    
    ## Note : HMM model is fit to particular trials but predicts posterior
    ## probabilities for transitions in all trials.
    
    # list all tastes
    taste_n = 30
    dig_in = hf5.list_nodes('/spike_trains')
    dig_in = [dig_in[i] if dig_in[i].__str__().find('dig_in')!=-1 else None for i in range(len(dig_in))]
    dig_in = list(filter(None, dig_in))
    
    on_trials = np.concatenate(tuple(np.where(dig_in[i].laser_durations[:] > 0)[0] + taste_n*i for i in range(len(dig_in))))

    post_prob = post_prob[on_trials,:,:]
    ## Since we're indexing post_prob by off_trials, EVERYTHING has to be indexed by off_trials
    
    sum_state_mass = np.sum(post_prob[:, indices, :], axis = (0,1)) #to find index of state with highest occurence during set time
    #transition = np.where(post_prob[:,:,np.argmax(sum_state_mass)] >= 0.8) #where[0] = x = trial, where[1] = y = time
    max_state = np.argmax(sum_state_mass)
    
    transition = []
    for i in range(post_prob.shape[0]):
        this_transition = np.where(post_prob[i, :, max_state] >= 0.8)[0]
        if len(this_transition) > 0:
            transition.append([i, this_transition[0]])
    
    transition_trials = np.array([transition[i][0] for i in range(len(transition))])
    ## Now EVERYTHING has to be indexed by transition trials as well -- MAYBE NOT
    fin_on_trials = on_trials[transition_trials]
    
            
    palatability_order = [3,4,1,2]
    trial_palatability = np.zeros(len(palatability_order)*30)
    for i in range(0,len(palatability_order)):
        trial_palatability[i*30:(i+1)*30] = np.repeat(palatability_order[i],30)[:]
    #trial_palatability = np.array(trial_palatability)
    
    # Get the spike array for each individual taste and merge all:
    spikes = np.concatenate(tuple(dig_in[i].spike_array[:] for i in range(len(dig_in))), axis = 0)
    on_spikes = spikes[on_trials,:,:]
    
    # Take spikes from trials with transitions and align them according to state transition per trial
    pre_stim_t = 2000   #time before stimulus
    data_window_pre = 500
    data_window_post = 750
    lined_spikes = []
    for trial, time in transition:
        lined_spikes.append(on_spikes[trial, :, pre_stim_t + t[time] - data_window_pre :pre_stim_t + t[time] + data_window_post])
    # lined_spikes shape = list of trials, every index = array[neurons , time]
    
    #for trial in transition[0]:
    #    for time in transition[1]:
    #        lined_spikes.append(spikes[trial, :, 1500 + t[time]:2500 + t[time]])
            
    lined_spikes = np.array(lined_spikes) # [trials, neurons, time]
    
    bin_window_size = 250
    step_size = 25
    ## Making PSTHs for the spikes
    lined_firing = np.zeros((lined_spikes.shape[0], lined_spikes.shape[1], int((data_window_pre + data_window_post - bin_window_size)/step_size) + 1))
    
    for i in range(lined_firing.shape[0]):
        for j in range(lined_firing.shape[1]):
            for k in range(lined_firing.shape[2]):
                lined_firing[i, j, k] = np.mean(lined_spikes[i, j, step_size*k:step_size*k + bin_window_size])
                
    test = lined_firing
    test = np.sum(test,0)
    for i in range(test.shape[0]):
        test[i,:] = test[i,:]/test[i,0]
      
    fig = plt.figure()
    plt.title('Aligned firing rates, On trials , Model states =' + str(states[num_states]))
    for i in range(test.shape[0]):
        plt.plot(test[i,:])
    fig.savefig(plot_dir + '/on_firing_states%i.png' % states[num_states])
    plt.close('all')
    
                
    #palatability = np.ones(lined_firing.shape[0])
    #palatability = np.array([trial_palatability[transition[i][0]] for i in range(0,len(transition))])
    palatability = trial_palatability[fin_on_trials]
    
    r_pearson = np.zeros((lined_firing.shape[1], lined_firing.shape[2]))
    p_pearson = np.ones((lined_firing.shape[1], lined_firing.shape[2]))
    
    for i in range(r_pearson.shape[0]):
        for j in range(r_pearson.shape[1]):
            r_pearson[i, j], p_pearson[i, j] = pearsonr(lined_firing[:, i, j], palatability)
            if np.isnan(r_pearson[i, j]):
                r_pearson[i, j] = 0.0
                p_pearson[i, j] = 1.0
    
    fig = plt.figure()
    plt.title('Correlation, On trials , Model states =' + str(states[num_states]))            
    plt.plot(np.mean(r_pearson[:, :]**2, axis = 0))
    fig.savefig(plot_dir + '/on_correlation_states%i.png' % states[num_states])
    plt.close('all')