#%%
import os
from pathlib import Path
script_dir = os.path.dirname(os.path.abspath(__file__)) # get the path of the current script
os.chdir(script_dir) # change the working directory
script_dir = Path(script_dir)

import numpy as np
import shutil

#%%
data_folder_name = 'one_focal'
data_folder = Path('/home/j/Desktop/hdd') / data_folder_name

# create folders if not exist
for folder_name in ['train', 'validation', 'test']:
    (data_folder / folder_name).mkdir(parents=True, exist_ok=True)

# move all files in train, validation, test to data_folder
for folder_name in ['train', 'validation', 'test']:
    for f in (data_folder / folder_name).iterdir():
        if f.is_file():
            shutil.move(str(f), str(data_folder / f.name))

#%%
# grab file names of simulation_results_*.npz and extract s1 values
simulation_results_file_names = list(data_folder.glob('simulation_results_*.npz'))
s1 = []
for f in simulation_results_file_names:
    stem = f.stem # e.g., 'simulation_results_123'
    parts = stem.replace('simulation_results_', '').split('_')
    s1.append(int(parts[0]))
s1 = np.array(s1)

# sort
sort_idx = np.argsort(s1)
s1 = s1[sort_idx]

n_files_to_use = len(s1)
s1 = s1[0:n_files_to_use]
n_samples = len(s1)

# randomly split into training, validation, and testing
perm = np.random.permutation(n_samples)
s1 = s1[perm]

n_train = int(0.8 * n_samples)
n_val = int(0.1 * n_samples)
n_test = n_samples - n_train - n_val

s1_train = s1[:n_train]
s1_validation = s1[n_train:n_train + n_val]
s1_test = s1[n_train + n_val:]

def move_files_to_folder(ids, folder_name, data_folder):
    for i in range(len(ids)):
        print(f'processing {i+1}/{len(ids)}')

        file_name = data_folder / f'simulation_results_{ids[i]}.npz'

        if file_name.exists():
            shutil.move(str(file_name), str(data_folder / folder_name / file_name.name))

#%%
# put files into folders
ids = s1_train
folder_name = 'train'
move_files_to_folder(ids, folder_name, data_folder)

ids = s1_validation
folder_name = 'validation'
move_files_to_folder(ids, folder_name, data_folder)

ids = s1_test
folder_name = 'test'
move_files_to_folder(ids, folder_name, data_folder)

print('done')
#%%
