import numpy as np
from pathlib import Path
import torch

def file_index(data_folder, n_files_to_use):
    # grab file names of simulation_results_*_*.npy and extract s1, s2 from filenames
    npy_files = list(data_folder.glob('simulation_results_*_*.npy'))
    s1 = []
    s2 = []
    for f in npy_files:
        # filename format: simulation_results_{s1}_{s2}.npy
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

    if n_files_to_use == -1:
        n_files_to_use = len(s1)
    s1 = s1[0:n_files_to_use]
    s2 = s2[0:n_files_to_use]

    return s1, s2

def input_output_data(start_idx, end_idx, data_folder, data_subfolder, s1_index, s2_index, non_e_id, parameters):
    # NOTE: 
    # the input argument 'non_e_id' has to be provided, because it is not necessary equal to parameters['non_e_id']
    # for example, when plotting mix rhythm activation time map, can set 'non_e_id' to an empty list to use all nodes

    x_temp = []
    y_temp = []
    for i in range(start_idx, end_idx):
        file_name_x = Path(data_subfolder) / f'simulation_results_{s1_index[i]}_{s2_index[i]}.npy'
        if parameters['data_flag'] == 0:
            action_potential = np.load(file_name_x, allow_pickle=True).item()['action_potential'] # shape (t, nodes)
            action_potential = (action_potential - np.min(action_potential)) / (np.max(action_potential) - np.min(action_potential)) # normalize to 0-1
            x = action_potential
            x[:, non_e_id] = 0
        elif parameters['data_flag'] == 1:
            electrogram_unipolar = np.load(file_name_x, allow_pickle=True).item()['electrogram_unipolar'] # shape (t, n_nodes)
            electrogram_unipolar = (electrogram_unipolar - np.min(electrogram_unipolar)) / (np.max(electrogram_unipolar) - np.min(electrogram_unipolar)) # normalize to 0-1
            x = electrogram_unipolar
            x[:, non_e_id] = 0
        
        x_temp.append(x)
        
        file_name_y_1 = Path(data_folder) / f'lat_{s1_index[i]}.npy'
        y_1 = np.load(file_name_y_1)
        y_1 = (y_1 - np.min(y_1)) / (np.max(y_1) - np.min(y_1)) # normalize to 0-1

        file_name_y_2 = Path(data_folder) / f'lat_{s2_index[i]}.npy'
        y_2 = np.load(file_name_y_2)
        y_2 = (y_2 - np.min(y_2)) / (np.max(y_2) - np.min(y_2)) # normalize to 0-1

        y = np.vstack((y_1, y_2)) # shape (2, nodes)
        y_temp.append(y)

    # stack into tensors
    input_data = torch.from_numpy(np.stack(x_temp, axis=0)) # shape (batch, t, n_node)
    output_data = torch.from_numpy(np.stack(y_temp, axis=0)) # shape (batch, 2, n_node)

    # grab time slices
    input_data = input_data[:, parameters['t_start']:parameters['t_end']:parameters['time_step'], :]

    input_data = input_data.float().to(parameters['device']) # ensure float32
    output_data = output_data.float().to(parameters['device']) # ensure float32

    return input_data, output_data
