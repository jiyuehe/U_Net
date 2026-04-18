import numpy as np
from pathlib import Path
import torch

def _normalize_to_unit_interval(values):
    min_value = np.min(values)
    max_value = np.max(values)
    range_value = max_value - min_value
    if range_value == 0:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - min_value) / range_value).astype(np.float32)

# def _extract_input_signal(payload, data_flag):
#     if data_flag == 0:
#         candidate_key = 'action_potential'
#     elif data_flag == 1:
#         candidate_key = 'electrogram_unipolar'
#     # else:
#     #     raise ValueError(f"Unsupported data_flag: {data_flag}. Expected 0 (action potential) or 1 (electrogram).")

#     # if candidate_key in payload:
#     return payload[candidate_key]

#     # available_keys = sorted(payload.keys())
#     # raise KeyError(
#     #     f"None of expected keys {candidate_keys} found in {file_path}. "
#     #     f"Available keys: {available_keys}"
#     # )

def file_index(data_folder, n_files_to_use):
    # grab file names of simulation results
    simulation_result_file_names = list(data_folder.glob('simulation_results_*.npz'))
    s1 = []
    for f in simulation_result_file_names:
        stem = f.stem # e.g., 'simulation_results_123'
        parts = stem.replace('simulation_results_', '').split('_')
        s1.append(int(parts[0]))
    s1 = np.array(s1)

    # sort
    sort_idx = np.argsort(s1)
    s1 = s1[sort_idx]

    N = len(s1)
    if n_files_to_use == -1:
        n_files_to_use = N
    
    idx = np.round(np.linspace(0, N - 1, n_files_to_use)).astype(int)
    s1 = s1[idx]

    return s1

def input_output_data(start_idx, end_idx, data_folder, data_subfolder, s1_index, s2_index, non_e_id, parameters):
    # NOTE: 
    # the input argument 'non_e_id' has to be provided, because it is not necessary equal to parameters['non_e_id']
    # for example, when plotting mix rhythm activation time map, can set 'non_e_id' to an empty list to use all nodes

    x_temp = []
    y_temp = []
    for i in range(start_idx, end_idx):
        if s2_index is not None:
            file_name_x = Path(data_subfolder) / f'simulation_results_{s1_index[i]}_{s2_index[i]}.npz'
        else:
            file_name_x = Path(data_subfolder) / f'simulation_results_{s1_index[i]}.npz'

        payload = dict(np.load(file_name_x, allow_pickle=False))
        
        if parameters['data_flag'] == 0:
            expected_key = 'action_potential_voxel3mm'
        elif parameters['data_flag'] == 1:
            expected_key = 'electrogram_unipolar'
        
        x = payload[expected_key]

        # x = _extract_input_signal(payload, parameters['data_flag'])
        x = _normalize_to_unit_interval(x)
        x[:, non_e_id] = 0
        
        x_temp.append(x)
        
        # file_name_y_1 = Path(data_folder) / f'lat_{s1_index[i]}.npz'
        # y_1 = np.load(file_name_y_1)['lat']
        y_1 = payload['lat']
        y_1 = _normalize_to_unit_interval(y_1)

        if s2_index is not None:
            file_name_y_2 = Path(data_folder) / f'lat_{s2_index[i]}.npz'
            y_2 = np.load(file_name_y_2)['lat']
            y_2 = _normalize_to_unit_interval(y_2)

            y = np.vstack((y_1, y_2)) # shape (2, nodes)
        else:
            y = y_1
        
        y_temp.append(y)

    # stack into tensors
    input_data = torch.from_numpy(np.stack(x_temp, axis=0)) # shape (batch, t, n_node)
    output_data = torch.from_numpy(np.stack(y_temp, axis=0)) # shape (batch, n_out_channel, n_node)

    # grab time slices
    input_data = input_data[:, parameters['t_start']:parameters['t_end']:parameters['time_step'], :]

    input_data = input_data.float().to(parameters['device']) # ensure float32
    output_data = output_data.float().to(parameters['device']) # ensure float32

    return input_data, output_data
