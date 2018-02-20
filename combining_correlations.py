#Better way to do all the try/except?
#Import dat ish
import numpy as np
import tables
import easygui
import os
import re
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plot_dir = easygui.diropenbox(msg = 'Choose directory to store the plots in')

# Ask the user for the hdf5 files that need to be plotted together
if easygui.ynbox('Load directories from file?', 'Title', ('Yes', 'No')):
    params_file = 'all_correlation_params'
    f = open(plot_dir + '/' + params_file, 'r')
    dirs = []
    for line in f.readlines():
        dirs.append(line.strip())
    f.close()
else:
    dirs = []
    while True:
        dir_name = easygui.diropenbox(msg = 'Choose a directory with a hdf5 file, hit cancel to stop choosing')
        try:
            if len(dir_name) > 0:    
                dirs.append(dir_name)
        except:
            break
        
    if easygui.ynbox('Save directories to plot folder?', 'Title', ('Yes', 'No')):
        if os.path.isfile(plot_dir + '/' + 'all_correlation_params'):
            if easygui.ynbox('Overwrite old parameters file?', 'Title', ('Yes', 'No')):
                f = open(plot_dir + '/' + 'all_correlation_params', 'w')
                for dir in dirs:
                    	print(dir, file=f)
                f.close()
        else:
                f = open(plot_dir + '/' + 'all_correlation_params', 'w')
                for dir in dirs:
                    	print(dir, file=f)
                f.close()

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

#Specify params or load from file
if easygui.ynbox('Load params from lining up file?', 'Title', ('Yes', 'No')):
    params_folder = easygui.diropenbox('Select parameters folder for lining up')
    params_file = 'correlation_lining_params'
    f = open(params_folder+'/'+params_file, 'r')
    params = []
    for line in f.readlines():
        params.append(line)
    f.close()
    pre_stim_t = int(params[7]) #2000   #time before stimulus
    data_window_pre = int(params[8]) #500   #window before transition time
    data_window_post = int(params[9]) #750  #window after transition time
    bin_window_size = int(params[10]) #250
    step_size = int(params[11]) #25
    
    
else:   #Request for parameters and save to file in correlation plot folder
    
    #Params for making PSTH's
    psth_params = easygui.multenterbox('Parameters for making PSTHs',fields = ['Pre-stimulus time (2000)','Time wanted before transition (500)','Time watned after transition (750)', 'Binning window size (250)', 'Window step size (25)'])
    pre_stim_t = int(psth_params[0]) #2000   #time before stimulus
    data_window_pre = int(psth_params[1]) #500   #window before transition time
    data_window_post = int(psth_params[2]) #750  #window after transition time
    bin_window_size = int(psth_params[3]) #250
    step_size = int(psth_params[4]) #25
 

# Make indexing vector to pull out correaltion from ancillary analysis array
params = easygui.multenterbox(fields = ['Unaligned correlation centering time (950)','Time length (7000)'])
middle_time = int(params[0]) #(700+1200)/2
t_length = int(params[1]) #7000

flank_window = int((data_window_pre + data_window_post - bin_window_size)/2)
time_vec = np.arange(0,t_length-bin_window_size+step_size,step_size)
index_vec = np.where((time_vec>=pre_stim_t+middle_time-flank_window)*(time_vec<=pre_stim_t+middle_time+flank_window))[0]

# Now run through the directories, and pull out the data
# Going to loop through each file and each model of the HMM (different number 
# of states) with on and off states separately


for num_state in unique_states:
    
    r_pearson_off = []
    r_pearson_off_scram = []
    del_transition_off = []
    #p_pearson_off = []
    
    r_pearson_on = []
    r_pearson_on_scram = []
    del_transition_on = []
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
        try:
            exec('r_pearson_off.append(hf5.root.spike_trains.multinomial_hmm_results.laser_off.states_%i.r_pearson[:])' % num_state)
        except:
            pass
        #exec('p_pearson_off.append(hf5.root.spike_trains.multinomial_hmm_results.laser_off.states_%i.p_pearson[:])' % num_state)
        try:
            exec('r_pearson_on.append(hf5.root.spike_trains.multinomial_hmm_results.laser_on.states_%i.r_pearson[:])' % num_state)
        except:
            pass
        #exec('p_pearson_on.append(hf5.root.spike_trains.multinomial_hmm_results.laser_on.states_%i.p_pearson[:])' % num_state)
        try:
            exec('del_transition_off.append(hf5.root.spike_trains.multinomial_hmm_results.laser_off.states_%i.palatability_state_transition_t[:])' % num_state)
        except:
            pass
        try:
            exec('del_transition_on.append(hf5.root.spike_trains.multinomial_hmm_results.laser_on.states_%i.palatability_state_transition_t[:])' % num_state)
        except:
            pass
        
        # Append r_pearson from the ancillary analysis of all files
        r_pearson_off_scram.append(np.ndarray.transpose(hf5.root.ancillary_analysis.r_pearson[0,:]))
        r_pearson_on_scram.append(np.ndarray.transpose(hf5.root.ancillary_analysis.r_pearson[1,:]))
        
        ## Also maintain a counter of the number of units in the analysis
        #exec('num_units_on += hf5.root.spike_trains.multinomial_hmm_results.laser_on.states_%i.p_pearson.shape[0]' % num_state)
        #exec('num_units_off += hf5.root.spike_trains.multinomial_hmm_results.laser_off.states_%i.p_pearson.shape[0]' % num_state)
        
        # Close the hdf5 file
        hf5.close()
            
            
    if len(unique_states)==1:
        r_pearson_off = r_pearson_off[0]
        #p_pearson_off = p_pearson_off[0]
        r_pearson_on = r_pearson_on[0]
        #p_pearson_on = p_pearson_on[0]
        r_pearson_off_scram = r_pearson_off_scram[0]
        r_pearson_on_scram = r_pearson_on_scram[0]
        del_transition_off = del_transition_off[0]
        del_transition_on = del_transition_on[0]
        
    else:
        r_pearson_off = np.concatenate(r_pearson_off,0)
        #p_pearson_off = np.concatenate(p_pearson_off,0)
        r_pearson_on = np.concatenate(r_pearson_on,0)
        #p_pearson_on = np.concatenate(p_pearson_on,0)
        r_pearson_off_scram = np.concatenate(r_pearson_off_scram,0)
        r_pearson_on_scram = np.concatenate(r_pearson_on_scram,0)
        del_transition_off = np.concatenate(del_transition_off,0)
        del_transition_on = np.concatenate(del_transition_on,0)
        
    #Index the unalgined correaltions by the index_vec
    r_pearson_off_scram = r_pearson_off_scram[:,index_vec]
    r_pearson_on_scram = r_pearson_on_scram[:,index_vec]
    
    # Sigmoid fitting
    def sigmoid(x, a, b, x_0, d):
        return (a/(1 + np.exp(-b*(x-x_0)))) + d
    x_data = np.arange(-data_window_pre,data_window_post-bin_window_size+step_size,step_size)
    
    #Returns x value at index corresponding to closest value in y
    def take_closest(val,y,x):
        est_median = np.min(y) + (np.max(y) - np.min(y))/2
        abs_dist = abs(y - est_median)
        closest_val_ind = np.where(abs_dist == np.min(abs_dist))[0]
        return x[closest_val_ind][0]
    
    dat_range_on = np.max(y_data_on)-np.min(y_data_on)
    mid_on = take_closest(dat_range_on/2,y_data_on,x_data)
    y_data_on = np.mean(r_pearson_on[:, :]**2, axis = 0)
    param_est_on = [dat_range_on,0,mid_on,np.min(y_data_on)]
    popt_on, pcov_on = curve_fit(sigmoid, x_data, y_data_on, p0 = param_est_on)
    
    dat_range_off = np.max(y_data_off)-np.min(y_data_off)
    mid_off = take_closest(dat_range_off/2,y_data_off,x_data)
    y_data_off = np.mean(r_pearson_off[:, :]**2, axis = 0)
    param_est_off = [dat_range_off,0,mid_off,np.min(y_data_off)]
    popt_off, pcov_off = curve_fit(sigmoid, x_data, y_data_off, p0 = param_est_off)
    
    dat_range_on_scram = np.max(y_data_on_scram)-np.min(y_data_on_scram)
    mid_on_scram = take_closest(dat_range_on_scram/2,y_data_on_scram,x_data)
    y_data_on_scram = np.mean(r_pearson_on_scram[:, :]**2, axis = 0)
    param_est_on_scram = [dat_range_on_scram,0,mid_on_scram,np.min(y_data_on_scram)]
    popt_on_scram, pcov_on_scram = curve_fit(sigmoid, x_data, y_data_on_scram, p0 = param_est_on_scram)
    
    dat_range_off_scram = np.max(y_data_off_scram)-np.min(y_data_off_scram)
    mid_off_scram = take_closest(dat_range_off_scram/2,y_data_off_scram,x_data)
    y_data_off_scram = np.mean(r_pearson_off_scram[:, :]**2, axis = 0)
    param_est_off_scram = [dat_range_off_scram,0,mid_off_scram,np.min(y_data_off_scram)]
    popt_off_scram, pcov_off_scram = curve_fit(sigmoid, x_data, y_data_off_scram, p0 = param_est_off_scram)

    #PLOTS!!!
    pearson_max = np.max(np.concatenate((np.mean(r_pearson_off[:, :]**2, axis = 0), np.mean(r_pearson_off_scram[:, :]**2, axis = 0))))
    pearson_min = np.min(np.concatenate((np.mean(r_pearson_off[:, :]**2, axis = 0), np.mean(r_pearson_off_scram[:, :]**2, axis = 0))))
    pearson_min_max = pearson_max - pearson_min  
    fig = plt.figure()
    plt.title('Correlation, Off trials , Model states =' + str(num_state))
    #Plot aligned correlation
    plt.errorbar(np.arange(-data_window_pre,data_window_post-bin_window_size+step_size,step_size),np.mean(r_pearson_off[:, :]**2, axis = 0),yerr=np.std(r_pearson_off[:,:]**2,axis = 0)/np.sqrt(r_pearson_off.shape[0]),label = 'Aligned')
    #Plot unaligned correlation
    plt.errorbar(np.arange(-data_window_pre,data_window_post-bin_window_size+step_size,step_size),np.mean(r_pearson_off_scram[:, :]**2, axis = 0),yerr=np.std(r_pearson_off_scram[:, :]**2,axis = 0)/np.sqrt(r_pearson_off_scram.shape[0]),label = 'Unaligned')
    fig.legend()
    plt.xlabel('State transition time')
    plt.ylabel('Pearson Correlation Coefficient')
    plt.plot(x_data, sigmoid(x_data, *popt_off), 'r-')
    plt.plot(x_data, sigmoid(x_data, *popt_off_scram), 'r-')
    ax = fig.add_subplot(111)
    ax.text(-400,pearson_min+(pearson_min_max*0.9),'Mean state transition \n %i +\- %i msec' % (np.mean(del_transition_off),np.std(del_transition_off)))
        
    fig.savefig(plot_dir + '/off_correlation_states%i.png' % num_state)
    plt.close('all')
    
    pearson_max = np.max(np.concatenate((np.mean(r_pearson_on[:, :]**2, axis = 0), np.mean(r_pearson_on_scram[:, :]**2, axis = 0))))
    pearson_min = np.min(np.concatenate((np.mean(r_pearson_on[:, :]**2, axis = 0), np.mean(r_pearson_on_scram[:, :]**2, axis = 0))))
    pearson_min_max = pearson_max - pearson_min  
    fig = plt.figure()
    plt.title('Correlation, On trials , Model states =' + str(num_state))
    #Plot aligned correlation
    plt.errorbar(np.arange(-data_window_pre,data_window_post-bin_window_size+step_size,step_size),np.mean(r_pearson_on[:, :]**2, axis = 0),yerr=np.std(r_pearson_on[:,:]**2,axis = 0)/np.sqrt(r_pearson_on.shape[0]), label = 'Aligned')
    #Plot unaligned correlation
    plt.errorbar(np.arange(-data_window_pre,data_window_post-bin_window_size+step_size,step_size),np.mean(r_pearson_on_scram[:, :]**2, axis = 0),yerr=np.std(r_pearson_on_scram[:, :]**2,axis = 0)/np.sqrt(r_pearson_on_scram.shape[0]), label = 'Unaligned')
    fig.legend()
    plt.xlabel('State transition time')
    plt.ylabel('Pearson Correlation Coefficient')
    plt.plot(x_data, sigmoid(x_data, *popt_on), 'r-')
    plt.plot(x_data, sigmoid(x_data, *popt_on_scram), 'r-')
    ax = fig.add_subplot(111)
    ax.text(-400,pearson_min+(pearson_min_max*0.9),'Mean state transition \n %i +\- %i msec' % (np.mean(del_transition_on),np.std(del_transition_on)))
       
    fig.savefig(plot_dir + '/on_correlation_states%i.png' % num_state)
    plt.close('all')
