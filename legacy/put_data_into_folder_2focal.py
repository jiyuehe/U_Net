# Copyright 2026 Jiyue He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#%%
import os
from pathlib import Path
script_dir = os.path.dirname(os.path.abspath(__file__)) # get the path of the current script
os.chdir(script_dir) # change the working directory
script_dir = Path(script_dir)

import numpy as np
import shutil

#%%
data_folder_name = '103_1-lagood_3mm'
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
# grab file names of simulation_results_*_*.npz and extract s1, s2 values
simulation_results_file_names = list(data_folder.glob('simulation_results_*_*.npz'))
s1 = []
s2 = []
for f in simulation_results_file_names:
    # filename format: simulation_results_{s1}_{s2}.npz
    stem = f.stem # e.g., 'simulation_results_123_456'
    parts = stem.replace('simulation_results_', '').split('_')
    s1.append(int(parts[0]))
    s2.append(int(parts[1]))
s1 = np.array(s1)
s2 = np.array(s2)

# sort by s1, apply same ordering to s2 (they are pairs)
sort_idx = np.argsort(s1)
s1 = s1[sort_idx]
s2 = s2[sort_idx]

n_files_to_use = len(s1)
s1 = s1[0:n_files_to_use]
s2 = s2[0:n_files_to_use]
n_samples = len(s1)

# randomly split into training, validation, and testing
perm = np.random.permutation(n_samples)
s1 = s1[perm]
s2 = s2[perm]

n_train = int(0.8 * n_samples)
n_val = int(0.1 * n_samples)
n_test = n_samples - n_train - n_val

s1_train = s1[:n_train]
s2_train = s2[:n_train]
s1_validation = s1[n_train:n_train + n_val]
s2_validation = s2[n_train:n_train + n_val]
s1_test = s1[n_train + n_val:]
s2_test = s2[n_train + n_val:]

def grab_file_names(id1, id2):
    # mix rhythm activation time
    file_name = f'lat_{id1}_{id2}.*'
    file_names_1 = list(data_folder.glob(file_name))

    # mix rhythm simulation results
    file_name = f'simulation_results_{id1}_{id2}.npz'
    file_names_2 = list(data_folder.glob(file_name))

    # combine all file_names lists into one variable
    file_names = [file_names_1, file_names_2]

    return file_names

def move_files_to_folder(id_1, id_2, folder_name, data_folder):
    for i in range(len(id_1)):
        print(f'processing {i+1}/{len(id_1)}')

        file_names = grab_file_names(id_1[i], id_2[i])
        folder = data_folder / folder_name
        for file_list in file_names:
            for f in file_list:
                if f.exists():
                    shutil.move(str(f), str(folder / f.name))

#%%
# put files into folders
id_1 = s1_train
id_2 = s2_train
folder_name = 'train'
move_files_to_folder(id_1, id_2, folder_name, data_folder)

id_1 = s1_validation
id_2 = s2_validation
folder_name = 'validation'
move_files_to_folder(id_1, id_2, folder_name, data_folder)

id_1 = s1_test
id_2 = s2_test
folder_name = 'test'
move_files_to_folder(id_1, id_2, folder_name, data_folder)

print('done')
#%%
