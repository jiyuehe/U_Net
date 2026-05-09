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

import os
from pathlib import Path
script_dir = os.path.dirname(os.path.abspath(__file__)) # get the path of the current script
os.chdir(script_dir) # change the working directory
script_dir = Path(script_dir)

# add the workspace root to Python path
from platform import node
import sys
workspace_root = Path().resolve().parent.parent # Path().resolve() returns an absolute path, the full path
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))
import PyHeartSim.common as common

import modules
import matplotlib.pyplot as plt 
import numpy as np

def plot_loss_history(result_folder):
    loss_history_path = result_folder / 'loss_history.txt'
    train_loss_history = []
    val_loss_history = []
    with open(loss_history_path, 'r') as f:
        for line in f:
            if line != 'train_loss\tval_loss\n':
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    train_loss_history.append(float(parts[0]))
                    val_loss_history.append(float(parts[1]))

    plt.figure()
    plt.plot(train_loss_history, 'r-', label='Training Loss')
    plt.plot(val_loss_history, 'b-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'r: train loss ({min(train_loss_history)*10**3:.2f}e-3), b: valid loss ({min(val_loss_history)*10**3:.2f}e-3)')
    plt.legend()
    plt.grid(True)
    plt.savefig(result_folder / 'loss_history.png', dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close()

def voxelize_nodes(node):
    min_coords = np.floor(node.min(axis=0)).astype(int) # node.min(axis=0) returns a 1D array containing the minimum x, y, and z values across all points
    max_coords = np.ceil(node.max(axis=0)).astype(int) # node.max(axis=0) returns a 1D array containing the maximum x, y, and z values across all points
    grid_shape = max_coords - min_coords + 1
    grid_indices = (node - min_coords).astype(int)
    
    voxels = np.zeros(grid_shape, dtype=bool)
    voxels[grid_indices[:,0], grid_indices[:,1], grid_indices[:,2]] = True

    return voxels, grid_indices

def scatter_or_voxel_plot(plot_scatter_voxel_flag, sparse_electrode_flag, node, e_id, non_e_id, voxels, grid_indices, converted_color, fig, ax):
    if plot_scatter_voxel_flag == 1:
        dpi = fig.dpi
        points_per_mm = dpi / 25.4
        if sparse_electrode_flag == 0:
            node_size = points_per_mm ** 2 * 2 * 2
        elif sparse_electrode_flag == 1:
            node_size = points_per_mm ** 2 * 2

        ax.scatter(node[non_e_id, 0], node[non_e_id, 1], node[non_e_id, 2], c='grey', s=1, edgecolor='none', linewidth=0, alpha=0.3)
        ax.scatter(node[e_id, 0], node[e_id, 1], node[e_id, 2], c=converted_color[e_id,:], edgecolor='none', linewidth=0, marker='s', s=node_size, depthshade=True)
    elif plot_scatter_voxel_flag == 0:
        voxel_color = np.zeros((*voxels.shape, 3))
        voxel_color[grid_indices[:,0], grid_indices[:,1], grid_indices[:,2]] = converted_color
        ax.voxels(voxels, facecolors=voxel_color, edgecolor=None, shade=False)

def plot_mix_rhythm_activation_time_map(sparse_electrode_flag, start_idx, end_idx, parameters):
    plot_scatter_voxel_flag = 0 # 1: scatter plot; 0: voxel plot

    # load input data
    if sparse_electrode_flag == 1:
        e_id = parameters['e_id']
        non_e_id = parameters['non_e_id']
    elif sparse_electrode_flag == 0:
        n_nodes = parameters['node'].shape[0]
        e_id = np.arange(n_nodes, dtype=np.int64)
        non_e_id = []

    input_data, _ = modules.load_data.input_output_data(start_idx, end_idx, parameters['data_folder'], parameters['data_folder'] / 'test', parameters['s1_test'], parameters['s2_test'], non_e_id, parameters)

    # voxelize the nodes (prepare grid for all samples)
    if parameters['geometry_flag'] in [1, 4]:
        node = parameters['node']
        voxels, grid_indices = voxelize_nodes(node)

    for sample_id in range(len(parameters['s1_test'])):
        # plot the mix rhythm map
        x = input_data[sample_id].cpu().numpy()

        if parameters['geometry_flag'] == 0:
            x = x.reshape((parameters['n_timepoints'], parameters['grid_height'] * parameters['grid_width'])) # shape: (t, nodes)

        # for each of the nodes, find the egm's max dvdt time index
        dvdt = np.diff(x, axis=0)
        if parameters['data_flag'] == 0: # action potential
            max_dvdt = dvdt  # positive derivative
        elif parameters['data_flag'] == 1: # electrogram
            max_dvdt = -dvdt  # negative derivative
        
        max_dvdt_indices = np.argmax(max_dvdt, axis=0)  # shape: (nodes,)
        max_dvdt_indices = max_dvdt_indices - np.min(max_dvdt_indices) # normalize to start from 0

        debug_plot = 1
        if debug_plot == 1:
            if sample_id == 0:
                node_id = e_id[0]
                plt.figure()
                plt.plot(x[:, node_id], 'b-')
                plt.plot(dvdt[:, node_id], 'r-')
                plt.axvline(x=max_dvdt_indices[node_id], color='k', linestyle='--')
                plt.title('example signal and its dv/dt')
                plt.xlabel('time index')
                plt.ylabel('amplitude')
                plt.legend(['signal', 'dv/dt', 'max dv/dt time'])
                plt.tight_layout()
                plt.savefig(parameters['result_folder'] / 'sample_egm.png', dpi=100, bbox_inches="tight", pad_inches=0)
                plt.close()

        data = max_dvdt_indices # this is the "local activation time" for the mix rhythm map
        data_min = np.nanmin(data)
        data_max = np.nanmax(data)
        data_threshold = data_min-0.1
        converted_color = common.convert_value_to_red_blue(data, data_min, data_max, data_threshold)
        converted_color[non_e_id,:] = 0.5 # set non-electrode nodes to grey

        fig = plt.figure(figsize=(8, 6), dpi=100)
        if parameters['geometry_flag'] == 0:
            color_image = converted_color.reshape((parameters['grid_height'], parameters['grid_width'], 3)) 
            plt.imshow(color_image, origin='lower', interpolation='nearest')
        elif parameters['geometry_flag'] in [1, 4]:
            ax = plt.axes(projection='3d')
            scatter_or_voxel_plot(plot_scatter_voxel_flag, sparse_electrode_flag, node, e_id, non_e_id, voxels, grid_indices, converted_color, fig, ax)
            common.set_axes_equal(ax)
            ax.view_init(elev=70, azim=-70)
        plt.axis('off')
        plt.tight_layout()

        if sparse_electrode_flag == 1:
            image_file_name = parameters['result_folder'] / f'{parameters["s1_test"][sample_id]}_{parameters["s2_test"][sample_id]}_lat_mix_sparse_nodes.png'
        elif sparse_electrode_flag == 0:
            image_file_name = parameters['result_folder'] / f'{parameters["s1_test"][sample_id]}_{parameters["s2_test"][sample_id]}_lat_mix_all_nodes.png'
        plt.savefig(image_file_name, dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close()
        common.crop_image.execute(image_file_name)

def plot_truth_and_predicted_activation_time_map(truth_data, predicted_data, file_names_test, parameters, data_type):
    plot_scatter_voxel_flag = 0 # 1: scatter plot; 0: voxel plot
    sparse_electrode_flag = 0 # color on all nodes

    n_samples = len(predicted_data)

    for sample_id in np.arange(n_samples):
        print(f'plotting sample {sample_id+1}/{n_samples}')

        # load geometry data
        if data_type == 'simulation':
            name_prefix = file_names_test[sample_id].split('_simulation_results_')[0]
        elif data_type == 'clinical':
            name_prefix = file_names_test[sample_id].split('_clinical_data')[0]

        data = np.load(parameters['data_folder_patient'] / f'{name_prefix}_clinical_data.npz', allow_pickle=True)
        map_data = {k: data[k] for k in data.files}
        node = map_data['voxel3mm_1mm_spacing']
        node = node - np.round(node.mean(axis=0)).astype(int) # shape (n_nodes, 3)
        n_nodes = node.shape[0]

        # electrode voxel ids
        if data_type == 'simulation':
            # randomly make some electrodes bad
            bad_electrode_ratio = 0.5
            n_bad_electrode = int(bad_electrode_ratio * n_nodes)
            non_e_id = np.random.choice(n_nodes, n_bad_electrode, replace=False)
        elif data_type == 'clinical':
            # find the good electrode nodes that have good signals
            act = map_data['clinical_activation_uni']
            good_e_id = [i for i, x in enumerate(act) if x != 0]
            non_e_id = np.setdiff1d(np.arange(n_nodes), good_e_id)

        e_id = np.arange(node.shape[0], dtype=np.int64)
        voxels, grid_indices = voxelize_nodes(node)

        data_truth = truth_data[sample_id].flatten()
        data_predicted = predicted_data[sample_id].flatten()

        # compute error
        error_mae = np.nanmean(np.abs(data_predicted - data_truth)) # mean absolute error on normalized activation time

        # plot true activation time map
        data_min = np.nanmin(data_truth)
        data_max = np.nanmax(data_truth)
        data_threshold = data_min-0.1
        converted_color = common.convert_value_to_red_blue(data_truth, data_min, data_max, data_threshold)

        # only color and show voxels with non-NaN data_truth
        valid_mask = ~np.isnan(data_truth)

        fig = plt.figure(figsize=(8, 6), dpi=100)
        ax = plt.axes(projection='3d')

        if plot_scatter_voxel_flag == 1:
            # Only plot non-NaN nodes
            ax.scatter(node[e_id[valid_mask]], node[e_id[valid_mask], 1], node[e_id[valid_mask], 2], c=converted_color[e_id[valid_mask], :], alpha=1.0, edgecolor='none', linewidth=0, marker='s', s=10, depthshade=True)
        else:
            # For voxel plot, set RGBA for all voxels, alpha=1 for valid, alpha=0.1 for NaN
            voxel_color = np.zeros((*voxels.shape, 4))  # RGBA
            # Set RGB for valid voxels
            voxel_color[grid_indices[valid_mask,0], grid_indices[valid_mask,1], grid_indices[valid_mask,2], :3] = converted_color[valid_mask]
            # Set alpha for valid voxels
            voxel_color[grid_indices[valid_mask,0], grid_indices[valid_mask,1], grid_indices[valid_mask,2], 3] = 1.0
            # Set alpha for NaN voxels (invalid_mask)
            invalid_mask = ~valid_mask
            if np.any(invalid_mask):
                # Optionally set color for NaN voxels (e.g., gray or leave as zero)
                voxel_color[grid_indices[invalid_mask,0], grid_indices[invalid_mask,1], grid_indices[invalid_mask,2], 3] = 0.05
            ax.voxels(voxels, facecolors=voxel_color, edgecolor=None, shade=False)
        common.set_axes_equal(ax)
        ax.view_init(elev=70, azim=-70)
        plt.axis('off')
        plt.tight_layout()

        image_file_name = parameters['result_folder'] / f'{file_names_test[sample_id][:-4]}_lat_truth.png'
        plt.savefig(image_file_name, dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close()
        common.crop_image(image_file_name)

        # plot predicted activation time map
        data_min = np.nanmin(data_predicted)
        data_max = np.nanmax(data_predicted)
        data_threshold = data_min-0.1
        converted_color = common.convert_value_to_red_blue(data_predicted, data_min, data_max, data_threshold)

        fig = plt.figure(figsize=(8, 6), dpi=100)
        ax = plt.axes(projection='3d')
        scatter_or_voxel_plot(plot_scatter_voxel_flag, sparse_electrode_flag, node, e_id, non_e_id, voxels, grid_indices, converted_color, fig, ax)
        common.set_axes_equal(ax)
        ax.view_init(elev=70, azim=-70)
        plt.axis('off')
        plt.tight_layout()

        image_file_name = parameters['result_folder'] / f'{file_names_test[sample_id][:-4]}_lat_predict_MAE_{error_mae:.4f}.png'
        plt.savefig(image_file_name, dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close()
        common.crop_image(image_file_name)
