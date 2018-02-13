#Import dat ish
import numpy as np
import tables
import easygui
import os
import re
import matplotlib.pyplot as plt

plot_dir = easygui.diropenbox(msg = 'Choose directory to store the plots in')

# Ask the user for the hdf5 files that need to be plotted together
dirs = []
while True:
    dir_name = easygui.diropenbox(msg = 'Choose a directory with a hdf5 file, hit cancel to stop choosing')
    try:
        if len(dir_name) > 0:    
            dirs.append(dir_name)
    except:
        break

unique_states = []
for dir_name in dirs:
        
    # Change to the directory
    os.chdir(dir_name)
    # Locate the hdf5 file
    file_list = os.listdir('./')
    hdf5_name = ''
    for files in file_list:
        if files[-2:] == 'h5':
            hdf5_name = files

    # Open the hdf5 file
    hf5 = tables.open_file(hdf5_name, 'r') 
    states = hf5.list_nodes('/spike_trains/multinomial_hmm_results/laser_off')
    states = re.findall('\d',states.__str__())
    for i in range(len(states)):
        states[i] = int(states[i])
    unique_states.append(states)

hf5.close()
unique_states = np.unique(unique_states)

# Make indexing vector to pull out correaltion from ancillary analysis array
window_size = 250
step_size = 25
t_length = 7000
time_vec = np.arange(0,t_length-window_size+step_size,step_size)
pre_stim_time = 2000
middle_time = (700+1200)/2
flank_window = 500
index_vec = np.where((time_vec>=pre_stim_time+middle_time-flank_window)*(time_vec<=pre_stim_time+middle_time+flank_window))[0]

# Now run through the directories, and pull out the data
# Going to loop through each file and each model of the HMM (different number 
# of states) with on and off states separately


for num_state in unique_states:
    
    r_pearson_off = []
    r_pearson_off_scram = []
    #p_pearson_off = []
    
    r_pearson_on = []
    r_pearson_on_scram = []
    #p_pearson_on = []
    
    num_units_on = 0
    num_units_off = 0
    
    for dir_name in dirs:
        
            # Change to the directory
            os.chdir(dir_name)
            # Locate the hdf5 file
            file_list = os.listdir('./')
            hdf5_name = ''
            for files in file_list:
                if files[-2:] == 'h5':
                    hdf5_name = files
        
            # Open the hdf5 file
            hf5 = tables.open_file(hdf5_name, 'r')
        
            # Pull the data from the /ancillary_analysis node
            exec('r_pearson_off.append(hf5.root.spike_trains.multinomial_hmm_results.laser_off.states_%i.r_pearson[:])' % num_state)
            #exec('p_pearson_off.append(hf5.root.spike_trains.multinomial_hmm_results.laser_off.states_%i.p_pearson[:])' % num_state)
            exec('r_pearson_on.append(hf5.root.spike_trains.multinomial_hmm_results.laser_on.states_%i.r_pearson[:])' % num_state)
            #exec('p_pearson_on.append(hf5.root.spike_trains.multinomial_hmm_results.laser_on.states_%i.p_pearson[:])' % num_state)
            
            # Append r_pearson from the ancillary analysis of all files
            r_pearson_off_scram.append(np.ndarray.transpose(hf5.root.ancillary_analysis.r_pearson[0,:]))
            r_pearson_on_scram.append(np.ndarray.transpose(hf5.root.ancillary_analysis.r_pearson[1,:]))
            ## Also maintain a counter of the number of units in the analysis
            exec('num_units_on += hf5.root.spike_trains.multinomial_hmm_results.laser_on.states_%i.p_pearson.shape[0]' % num_state)
            exec('num_units_off += hf5.root.spike_trains.multinomial_hmm_results.laser_off.states_%i.p_pearson.shape[0]' % num_state)
    
            # Close the hdf5 file
            hf5.close()
            
    if len(unique_states)==1:
        r_pearson_off = r_pearson_off[0]
        #p_pearson_off = p_pearson_off[0]
        r_pearson_on = r_pearson_on[0]
        #p_pearson_on = p_pearson_on[0]
        r_pearson_off_scram = r_pearson_off_scram[0]
        r_pearson_on_scram = r_pearson_on_scram[0]
    else:
        r_pearson_off = np.concatenate(r_pearson_off,0)
        #p_pearson_off = np.concatenate(p_pearson_off,0)
        r_pearson_on = np.concatenate(r_pearson_on,0)
        #p_pearson_on = np.concatenate(p_pearson_on,0)
        r_pearson_off_scram = np.concatenate(r_pearson_off_scram,0)
        r_pearson_on_scram = np.concatenate(r_pearson_on_scram,0)
        
    #Index the unalgined correaltions by the index_vec
    r_pearson_off_scram = r_pearson_off_scram[:,index_vec]
    r_pearson_on_scram = r_pearson_on_scram[:,index_vec]

        
    fig = plt.figure()
    plt.title('Correlation, Off trials , Model states =' + str(num_state))
    #Plot aligned correlation
    #aligned, = plt.plot(np.mean(r_pearson_off[:, :]**2, axis = 0))
    plt.errorbar(range(r_pearson_off.shape[1]),np.mean(r_pearson_off[:, :]**2, axis = 0),yerr=np.std(r_pearson_off[:,:]**2,axis = 0)/np.sqrt(r_pearson_off.shape[0]),label = 'Aligned')
    #Plot unaligned correlation
    #unaligned, = plt.plot(np.mean(r_pearson_off_scram[:, :]**2, axis = 0)
    plt.errorbar(range(r_pearson_off_scram.shape[1]),np.mean(r_pearson_off_scram[:, :]**2, axis = 0),yerr=np.std(r_pearson_off_scram[:, :]**2,axis = 0)/np.sqrt(r_pearson_off_scram.shape[0]),label = 'Unaligned')
    fig.legend()
    fig.savefig(plot_dir + '/off_correlation_states%i.png' % num_state)
    plt.close('all')
    
    fig = plt.figure()
    plt.title('Correlation, On trials , Model states =' + str(num_state))
    #Plot aligned correlation
    #plt.plot(np.mean(r_pearson_on[:, :]**2, axis = 0))
    plt.errorbar(range(r_pearson_on.shape[1]),np.mean(r_pearson_on[:, :]**2, axis = 0),yerr=np.std(r_pearson_on[:,:]**2,axis = 0)/np.sqrt(r_pearson_on.shape[0]), label = 'Aligned')
    #Plot unaligned correlation
    #plt.plot(np.mean(r_pearson_on_scram[:, :]**2, axis = 0))
    plt.errorbar(range(r_pearson_on_scram.shape[1]),np.mean(r_pearson_on_scram[:, :]**2, axis = 0),yerr=np.std(r_pearson_on_scram[:, :]**2,axis = 0)/np.sqrt(r_pearson_on_scram.shape[0]), label = 'Unaligned')
    fig.legend()
    fig.savefig(plot_dir + '/on_correlation_states%i.png' % num_state)
    plt.close('all')