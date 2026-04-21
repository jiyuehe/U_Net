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

def input_output_data(start_idx, end_idx, data_folder, data_subfolder, s1_index, s2_index, non_e_id, parameters):
    # NOTE: 
    # the input argument 'non_e_id' has to be provided, because it is not necessary equal to parameters['non_e_id']
    # for example, when plotting mix rhythm activation time map, can set 'non_e_id' to an empty list to use all nodes

    x_temp = []
    y_temp = []
    for i in range(start_idx, end_idx):
        file_name_x = Path(data_subfolder) / f'simulation_results_{s1_index[i]}.npz'
        
        payload = dict(np.load(file_name_x, allow_pickle=False))
        
        if parameters['data_flag'] == 0:
            expected_key = 'action_potential_voxel3mm'
        elif parameters['data_flag'] == 1:
            expected_key = 'electrogram_unipolar'
        
        x = payload[expected_key]
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
