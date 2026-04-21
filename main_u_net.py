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

import modules

import torch
import numpy as np
import matplotlib.pyplot as plt 
from torchview import draw_graph # for visualizing the neural network model architecture

#%matplotlib tk 
# make the Matplotlib plot pop up in a window instead of inline in the Jupyter notebook when debugging; change to %matplotlib inline if want to show plots in the notebook

#%%
# parameters
parameters = {}

# time samples
parameters['t_start'] = 0
parameters['time_step'] = 1
parameters['n_timepoints'] = 1000
parameters['t_end'] = parameters['t_start'] + parameters['n_timepoints'] * parameters['time_step']

# training parameters
parameters['batch_size'] = 32 # number of training samples (electrograms-activation_maps pairs) are processed together in one pass during training
parameters['learning_rate'] = 1e-4 # too small or too big are both bad
parameters['epochs'] = 100 # maximum epochs (training may stop earlier with early stopping)
parameters['early_stopping_patience'] = 6 # stop training if no improvement for this many epochs

# data parameters
parameters['geometry_flag'] = 1 # 1: patient 3D atrium, 0: 2D sheet

# mode settings
train_flag = 1 # 1: will train the model; 0: only do prediction with the pre-trained model
continue_training = 0 # 1: load best_unet_model.pth and continue training; 0: train from scratch
testing_data_flag = 0 # 0: simulation data; 1: clinical data
data_type = '1focal' # '1focal' or '2focal'

# geometry
if parameters['geometry_flag'] == 0:
   map_file_name = ''
   data_folder_name = ''
   parameters['result_folder'] = ''
   parameters['grid_height'] = 128 # do not change
   parameters['grid_width'] = 128 # do not change
elif parameters['geometry_flag'] == 1:
   parameters['data_folder'] = Path('/home/j/Desktop/hdd/simulation_results')
   parameters['result_folder'] = script_dir / 'result'
   parameters['result_folder'].mkdir(exist_ok=True)
   
   parameters['grid_height'] = [] # unused; for code compatibility
   parameters['grid_width'] = [] # unused; for code compatibility

#%%
train_validation_test_file_names = 'train_validation_test_file_names.txt'
if not (parameters['data_folder'] / train_validation_test_file_names).exists(): # if file not exist
   # grab file names of the simulation results
   simulation_files = list(parameters['data_folder'].glob('*.npz'))

   n_samples = len(simulation_files)

   # randomly split into training, validation, and testing
   perm = np.random.permutation(n_samples)

   n_train = int(0.8 * n_samples)
   n_val = int(0.1 * n_samples)
   n_test = n_samples - n_train - n_val

   file_id_train = perm[:n_train]
   file_id_validation = perm[n_train:n_train + n_val]
   file_id_test = perm[n_train + n_val:]

   # write a text file to save the file names of the training, validation and test data 
   with open(parameters['data_folder'] / train_validation_test_file_names, 'w') as f:
      f.write('train:\n')
      for idx in file_id_train:
         f.write(f'{simulation_files[idx].name}\n')
      f.write('validation:\n')
      for idx in file_id_validation:
         f.write(f'{simulation_files[idx].name}\n')
      f.write('test:\n')
      for idx in file_id_test:
         f.write(f'{simulation_files[idx].name}\n')


















#%%
# create the U-Net model
if parameters['geometry_flag'] == 1:
   try:
      import MinkowskiEngine as ME # https://nvidia.github.io/MinkowskiEngine/overview.html
   except ImportError:
      print('MinkowskiEngine is not installed.')
      pass

parameters['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if parameters['geometry_flag'] == 0:
   parameters['model'] = modules.unet.UNet(in_channels=parameters['n_timepoints'], out_channels=2).to(parameters['device'])
elif parameters['geometry_flag'] == 1:
   try: 
      parameters['model'] = modules.unet_minkowski.MinkowskiUNet(in_channels=parameters['n_timepoints']+1, out_channels=1,D=3).to(parameters['device']) 
      # D is the dimension of the input data
      # in_channels is n_timepoints channels for the electrogram, plus 1 channel as an indicator for electrode nodes (1 for electrode nodes, 0 for non-electrode nodes)
      # out_channels is 1 for the predicted activation time map
   except Exception as e:
      print('MinkowskiEngine is not installed.')
      pass

debug_flag = 0
if debug_flag == 1:
   print(f'Model created with {sum(p.numel() for p in parameters["model"].parameters())} parameters')

   if parameters['geometry_flag'] == 0:
      model_graph = draw_graph(
         parameters['model'],
         input_size=(parameters['batch_size'], parameters['n_timepoints'], parameters['grid_height'], parameters['grid_width']),
         graph_dir='TB',             
         roll=True, # hide internal ops
      )
      g = model_graph.visual_graph
      g.attr(
         dpi="300",
         fontname="Helvetica",
         fontsize="24",
         ranksep="0.5", # spacing between layers
         nodesep="0.5", # spacing between nodes
      )
      model_graph.visual_graph.render(parameters['result_folder'] / 'unet_torchview', format='png', cleanup=True)
   elif parameters['geometry_flag'] == 1:
      # torchview does not support MinkowskiEngine SparseTensor
      # use print to show model architecture instead
      print(parameters['model'])

#%%
# load geometry
data = np.load(map_file_name, allow_pickle=True)
map_data = {k: data[k] for k in data.files}

# find the good electrode nodes that have good signals
voxel3mm_id_of_electrode = map_data['voxel3mm_id_of_electrode']
act = map_data['activation_uni']
good_id = [i for i, x in enumerate(act) if x != 0]
good_e_id = voxel3mm_id_of_electrode[good_id]

voxel3mm_1mm_spacing = map_data['voxel3mm_1mm_spacing']
voxel3mm_1mm_spacing = voxel3mm_1mm_spacing - np.round(voxel3mm_1mm_spacing.mean(axis=0)).astype(int)

parameters['node'] = voxel3mm_1mm_spacing
n_nodes = parameters['node'].shape[0]

n_electrode = len(good_e_id)
coef = n_electrode / n_nodes
print(f'n_node: {n_nodes}, n_electrode: {n_electrode}, percentage: {coef*100:.2f}%')

parameters['e_id'] = good_e_id
parameters['non_e_id'] = np.setdiff1d(np.arange(n_nodes), parameters['e_id'])

debug_plot = 0
if debug_plot == 1:
   # use Matplotlib here because I did not and do not want to install plotly in the MinkowskiEngine docker container 
   node = parameters['node']

   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')
   ax.scatter(node[:, 0], node[:, 1], node[:, 2], s=1, c='gray', alpha=0.7)
   ax.scatter(node[good_e_id, 0], node[good_e_id, 1], node[good_e_id, 2], s=9, c='blue')
   ax.set_axis_off()
   plt.tight_layout()
   plt.show()

#%%
if parameters['geometry_flag'] == 0:
   parameters['data_folder'] = Path('/home/j/Desktop/hdd') / data_folder_name
elif parameters['geometry_flag'] == 1:
   parameters['data_folder'] = Path('/data') / data_folder_name # this is when using the MinkowskiEngine docker container

# load data file index
data_type = '1focal' # '1focal' or '2focal'
if data_type == '1focal':
   # load training data file index
   n_files_to_use = -1 # -1: use all files; or specify a number
   parameters['s1_train'] = modules.load_data_1focal.file_index(parameters['data_folder'] / 'train', n_files_to_use)

   # load validation data file index
   n_files_to_use = -1 # -1: use all files; or specify a number
   parameters['s1_validation'] = modules.load_data_1focal.file_index(parameters['data_folder'] / 'validation', n_files_to_use)

   # load test data file index
   n_files_to_use = 10 # -1: use all files; or specify a number
   parameters['s1_test'] = modules.load_data_1focal.file_index(parameters['data_folder'] / 'test', n_files_to_use)

print(f'n_train: {len(parameters["s1_train"])}, n_validation: {len(parameters["s1_validation"])}, n_test: {len(parameters["s1_test"])}')

#%%
# train the model
if train_flag == 1:
   print('train model')

   if continue_training == 1: # load pre-trained model to continue training
      model_path = parameters['result_folder'] / 'best_unet_model.pth'
      print(f'loading pre-trained model from {model_path}')
      parameters['model'].load_state_dict(torch.load(model_path, map_location=parameters['device']))

   # train the model
   train_loss_history, val_loss_history = modules.train_predict.train_model(parameters)

# plot loss history
modules.result_analysis_1focal.plot_loss_history(parameters['result_folder'], parameters['s1_train'])

#%%
if train_flag == 0:
   # predict with test data
   print('model prediction')

   parameters['model'].load_state_dict(torch.load(parameters['result_folder'] / 'best_unet_model.pth', map_location=parameters['device'])) # load the best model

   if testing_data_flag == 0:
      predicted_data, truth_data = modules.train_predict.predict(parameters)
   elif testing_data_flag == 1:
      parameters['model'].eval()

      n_test_samples = 1
      n_test_batches = (n_test_samples + parameters['batch_size'] - 1) // parameters['batch_size']

      ##########
      # file_path = script_dir.parent / '0_data' / 'simulation_results_6890_20931.npy'
      # simulation_data = np.load(file_path, allow_pickle=True).item()
      # simulation_egm = simulation_data['electrogram_unipolar']

      all_predictions = []
      all_truths = []
      with torch.no_grad():
         for batch_idx in range(n_test_batches):
            print(f'  Prediction batch {batch_idx+1}/{n_test_batches}')

            start_idx = batch_idx * parameters['batch_size']
            end_idx = min((batch_idx + 1) * parameters['batch_size'], n_test_samples)

            n_node = parameters['node'].shape[0]
            x_temp = []
            for i in range(start_idx, end_idx):
               ##########
               # electrogram_unipolar = map_data['clinical_electrogram_unipolar_refined'].T # shape (t, n_nodes)
               electrogram_unipolar = map_data['clinical_electrogram_unipolar'].T # shape (t, n_nodes)
               # electrogram_unipolar = simulation_egm
               electrogram_unipolar = (electrogram_unipolar - np.min(electrogram_unipolar)) / (np.max(electrogram_unipolar) - np.min(electrogram_unipolar)) # normalize to 0-1
               
               x = np.zeros((1000, n_node)) # assign all nodes zero signal

               ##########
               e_id = map_data['voxel3mm_id_of_electrode']
               x[:, e_id] = electrogram_unipolar[2000-500:2000+500, :] # assign electrode nodes the electrogram signal
               # x[:, e_id] = simulation_egm[:, e_id]
               
               x_temp.append(x)
            
            # stack into tensors
            input_data = torch.from_numpy(np.stack(x_temp, axis=0)) # shape (batch, t, n_node)
            # output_data = torch.from_numpy(np.stack(y_temp, axis=0)) # shape (batch, 2, n_node)

            # grab time slices
            input_data = input_data[:, parameters['t_start']:parameters['t_end']:parameters['time_step'], :]

            input_data = input_data.float().to(parameters['device']) # ensure float32
            # output_data = output_data.float().to(parameters['device']) # ensure float32

            device = parameters['device']
            node = parameters['node']
            # create nodes_batch for MinkowskiEngine: shape (N_total, 4) where each row is [batch_idx, x, y, z]
            # node has shape (n_nodes, 3)
            nodes_list = []
            current_batch_size = input_data.shape[0]
            for b in range(current_batch_size):
               n_nodes = node.shape[0]

               batch_indices = torch.full((n_nodes, 1), b, dtype=torch.int32)
               sample_nodes = torch.cat([batch_indices, torch.from_numpy(node).int()], dim=1) # convert xyz to integers. shape (n_nodes, 4)
               nodes_list.append(sample_nodes)
            nodes_batch = torch.cat(nodes_list, dim=0).to(device)  # (batch * n_nodes, 4)

            # reshape input data: (batch, t, nodes) -> (batch * nodes, t)
            feats_batch = input_data.permute(0, 2, 1).reshape(-1, input_data.shape[1])
            
            # create MinkowskiEngine sparse tensor
            neural_network_input = ME.SparseTensor(features=feats_batch, coordinates=nodes_batch, device=device)
            # target_sparse = ME.SparseTensor(features=targets_batch, coordinates=nodes_batch, device=device)

            # forward pass
            outputs = parameters['model'](neural_network_input)

            current_batch_size = input_data.shape[0]

            # convert to dense tensor: shape (batch, C, X, Y, Z) for 3D
            # find the minimum coordinate for dense conversion (required if any coordinate is negative)
            min_coord = torch.IntTensor(np.array(parameters['node']).min(axis=0).flatten())
            
            # extract predictions at shifted coordinates
            dense = outputs.dense(min_coordinate=min_coord)
            prediction_dense = dense[0].cpu()  # shape: (batch, 2, X, Y, Z)
            n_nodes = parameters['node'].shape[0]
            shifted_coord = np.array(parameters['node']).astype(int) - min_coord.numpy() # shift node by min_coord for correct indexing
            prediction = np.zeros((current_batch_size, 2, n_nodes), dtype=np.float32)
            for b in range(current_batch_size):
               for n, (x, y, z) in enumerate(shifted_coord):
                  prediction[b, :, n] = prediction_dense[b, :, x, y, z]
            prediction = torch.tensor(prediction)

            all_predictions.append(prediction)

         # concatenate all batches
         predicted_data = torch.cat(all_predictions, dim=0).numpy()

   # save the prediction results
   np.save(parameters['result_folder'] / f'predictions_{name_prefix}.npy', predicted_data)

   #%%
   if testing_data_flag == 0:
      # plot mix rhythm activation time map
      start_idx = 0
      end_idx = len(parameters['s1_test'])

      if data_type == '2focal':
         # plot full mix rhythm data
         sparse_electrode_flag = 0 # 1: use sparse electrode nodes; 0: use all nodes
         modules.result_analysis.plot_mix_rhythm_activation_time_map(sparse_electrode_flag, start_idx, end_idx, parameters)

         # plot sparse electrode nodes mix rhythm data
         sparse_electrode_flag = 1 # 1: use sparse electrode nodes; 0: use all nodes
         modules.result_analysis.plot_mix_rhythm_activation_time_map(sparse_electrode_flag, start_idx, end_idx, parameters)

         # plot truth and predicted activation time map
         modules.result_analysis.plot_truth_and_predicted_activation_time_map(truth_data, predicted_data, parameters)
      elif data_type == '1focal':
         modules.result_analysis_1focal.plot_truth_and_predicted_activation_time_map(truth_data, predicted_data, parameters)

print('done')

#%%
