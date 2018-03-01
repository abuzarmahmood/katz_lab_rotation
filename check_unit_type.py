## Check neuron type for given analysis
import tables
import os
import easygui
import numpy as np
data_direc = easygui.diropenbox('Choose directory to pick hdf5 file from')
os.chdir(data_direc)

# Look for the hdf5 file in the directory
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
	if files[-2:] == 'h5':
		hdf5_name = files

# Open the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r+')
unit_descrip = hf5.root.unit_descriptor[:]

# Read used units file
f = open('blech.hmm_units')
hmm_units = []
for line in f.readlines():
    hmm_units.append(int(line.strip()))
f.close()

hmm_units = np.array(hmm_units)
selected_units = unit_descrip[hmm_units]
unit_val = []
for i in range(len(selected_units)):
    unit_val.append(selected_units[i]['single_unit'])
print(np.mean(unit_val))
hf5.close()