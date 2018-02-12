# Add saving parameters to file

####################################
## Lining up by state transitions ##
####################################

# Import libraries, change folder, read file, make plot directory
import os
import tables
import numpy as np
import pylab as plt
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import re
import easygui

data_direc = easygui.diropenbox()
os.chdir(data_direc)

# Look for the hdf5 file in the directory
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
	if files[-2:] == 'h5':
		hdf5_name = files

# Open the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r+')

plot_dir = './correlation_analysis_plots'
if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)

# For extracting spike trains
dig_in = hf5.list_nodes('/spike_trains')
dig_in = [dig_in[i] if dig_in[i].__str__().find('dig_in')!=-1 else None for i in range(len(dig_in))]
dig_in = list(filter(None, dig_in))

## Number of trials per taste for indexing when concatenating spike trains
taste_n = [dig_in[i].spike_array[:].shape[0] for i in range(len(dig_in))]
if np.std(taste_n) == 0:
    taste_n = taste_n[0]
else:
    taste_n = int(easygui.multenterbox('How many trails per taste??',fields = ['# of trials'])[0])

lining_params = easygui.multenterbox('Parameters for correlation',fields = ['Start time for palatability', 'End time for palatability', 'Probability to treat state as significant'])
min_t = int(lining_params[0])
max_t = int(lining_params[1])
state_sig = float(lining_params[2])

################
## OFF TRIALS ##
################
# Find number of states to use in extracting from hdf5 file
states = hf5.list_nodes('/spike_trains/multinomial_hmm_results/laser_off')

states = re.findall('\d',states.__str__())
for i in range(len(states)):
    states[i] = int(states[i])


for num_states in range(len(states)): ## Loop through all models in HMM
    exec('post_prob = hf5.root.spike_trains.multinomial_hmm_results.laser_off.states_%i.posterior_proba[:]' % states[num_states])
    #post_prob shape = [trials,time,states]
    exec('t = np.array(hf5.root.spike_trains.multinomial_hmm_results.laser_off.states_%i.time[:])' % states[num_states])

    indices = np.where((t >= min_t)*(t <= max_t))[0]

    ## Note : HMM model is fit to particular trials but predicts posterior
    ## probabilities for transitions in all trials.
    ## Only using trials for which the model was fit i.e off_trials for off model and on_trials for on model.

    # Which trials did NOT have laser
    off_trials = np.concatenate(tuple(np.where(dig_in[i].laser_durations[:] == 0)[0] + taste_n*i for i in range(len(dig_in))))

    post_prob = post_prob[off_trials,:,:]
    ## Since we're indexing post_prob by off_trials, EVERYTHING has to be indexed by off_trials

    sum_state_mass = np.sum(post_prob[:, indices, :], axis = (0,1)) #to find index of state with highest occurence during set time
    #transition = np.where(post_prob[:,:,np.argmax(sum_state_mass)] >= 0.8) #where[0] = x = trial, where[1] = y = time
    max_state = np.argmax(sum_state_mass)

    # Find which trials have transitions and first point of transition
    transition = []
    for i in range(post_prob.shape[0]):
        this_transition = np.where(post_prob[i, :, max_state] >= state_sig)[0]
        if len(this_transition) > 0:
            transition.append([i, this_transition[0]])

    transition_trials = np.array([transition[i][0] for i in range(len(transition))]) # For indexing
    fin_off_trials = off_trials[transition_trials] # Since we're subsetting off trials and then only trials with transitions


    palatability_order = [3,4,1,2] #Manually entered, corresponds to dig_in [0, 1, 2, 3]

    #Make palatability vector for ALL trials
    trial_palatability = np.zeros(len(palatability_order)*taste_n)
    for i in range(0,len(palatability_order)):
        trial_palatability[i*taste_n:(i+1)*taste_n] = np.repeat(palatability_order[i],taste_n)[:]

    # Get the spike array for each individual taste and merge all:
    spikes = np.concatenate(tuple(dig_in[i].spike_array[:] for i in range(len(dig_in))), axis = 0)
    off_spikes = spikes[off_trials,:,:] #Index trials with no laser

    # Take spikes from trials with transitions and align them according to state transition per trial
    pre_stim_t = 2000   #time before stimulus
    data_window_pre = 500   #window before transition time
    data_window_post = 750  #window after transition time
    lined_spikes = []
    for trial, time in transition:
        lined_spikes.append(off_spikes[trial, :, pre_stim_t + t[time] - data_window_pre :pre_stim_t + t[time] + data_window_post])
    # lined_spikes shape = list of trials, every index = array[neurons , time]
    lined_spikes = np.array(lined_spikes) # [trials, neurons, time]

    bin_window_size = 250
    step_size = 25
    ## Making PSTHs for the spikes
    lined_firing = np.zeros((lined_spikes.shape[0], lined_spikes.shape[1], int((data_window_pre + data_window_post - bin_window_size)/step_size) + 1))

    for i in range(lined_firing.shape[0]):
        for j in range(lined_firing.shape[1]):
            for k in range(lined_firing.shape[2]):
                lined_firing[i, j, k] = np.mean(lined_spikes[i, j, step_size*k:step_size*k + bin_window_size])

    # Plot normalized PSTH of aligned spikes (just for visual inspection)
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

    # Palatability coefficient used for analysis
    palatability = trial_palatability[fin_off_trials]

    # Calculate Pearson parameters and make figures
    r_pearson = np.zeros((lined_firing.shape[1], lined_firing.shape[2]))
    p_pearson = np.ones((lined_firing.shape[1], lined_firing.shape[2]))

    for i in range(r_pearson.shape[0]):
        for j in range(r_pearson.shape[1]):
            r_pearson[i, j], p_pearson[i, j] = pearsonr(lined_firing[:, i, j], palatability)
            if np.isnan(r_pearson[i, j]):
                r_pearson[i, j] = 0.0
                p_pearson[i, j] = 1.0

    # Delete the r_pearson and p_pearson nodes if they already exist
    try:
        hf5.remove_node('/spike_trains/multinomial_hmm_results/laser_off/states_%i' % states[num_states], 'r_pearson')
        hf5.remove_node('/spike_trains/multinomial_hmm_results/laser_off/states_%i' % states[num_states], 'p_pearson')
    except:
        pass

    hf5.create_array('/spike_trains/multinomial_hmm_results/laser_off/states_%i' % states[num_states], 'r_pearson', r_pearson)
    hf5.create_array('/spike_trains/multinomial_hmm_results/laser_off/states_%i' % states[num_states], 'p_pearson', p_pearson)
    hf5.flush()

    fig = plt.figure()
    plt.title('Correlation, Off trials , Model states =' + str(states[num_states]))
    plt.plot(np.mean(r_pearson[:, :]**2, axis = 0))
    fig.savefig(plot_dir + '/off_correlation_states%i.png' % states[num_states])
    plt.close('all')

###############
## ON TRIALS ##
###############
    # Code here is same as that for off trials - please refer to comments there for any questions - Thx! (Abu)
for num_states in range(len(states)):
    ## Loop through states
    exec('post_prob = hf5.root.spike_trains.multinomial_hmm_results.laser_on.states_%i.posterior_proba[:]' % states[num_states]) #shape = [trials,time,states]
    exec('t = np.array(hf5.root.spike_trains.multinomial_hmm_results.laser_on.states_%i.time[:])' % states[num_states])
    indices = np.where((t >= min_t)*(t <= max_t))[0]

    ## Note : HMM model is fit to particular trials but predicts posterior
    ## probabilities for transitions in all trials.

    # list all tastes
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
        this_transition = np.where(post_prob[i, :, max_state] >= state_sig)[0]
        if len(this_transition) > 0:
            transition.append([i, this_transition[0]])

    transition_trials = np.array([transition[i][0] for i in range(len(transition))])
    ## Now EVERYTHING has to be indexed by transition trials as well -- MAYBE NOT
    fin_on_trials = on_trials[transition_trials]


    palatability_order = [3,4,1,2]
    trial_palatability = np.zeros(len(palatability_order)*taste_n)
    for i in range(0,len(palatability_order)):
        trial_palatability[i*taste_n:(i+1)*taste_n] = np.repeat(palatability_order[i],taste_n)[:]
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

    # Delete the r_pearson and p_pearson nodes if they already exist
    try:
        hf5.remove_node('/spike_trains/multinomial_hmm_results/laser_off/states_%i' % states[num_states], 'r_pearson')
        hf5.remove_node('/spike_trains/multinomial_hmm_results/laser_off/states_%i' % states[num_states], 'p_pearson')
    except:
        pass

    hf5.create_array('/spike_trains/multinomial_hmm_results/laser_off/states_%i' % states[num_states], 'r_pearson', r_pearson)
    hf5.create_array('/spike_trains/multinomial_hmm_results/laser_off/states_%i' % states[num_states], 'p_pearson', p_pearson)
    hf5.flush()

    fig = plt.figure()
    plt.title('Correlation, On trials , Model states =' + str(states[num_states]))
    plt.plot(np.mean(r_pearson[:, :]**2, axis = 0))
    fig.savefig(plot_dir + '/on_correlation_states%i.png' % states[num_states])
    plt.close('all')

hf5.close()
