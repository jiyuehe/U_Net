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
parameters['n_timepoints'] = 500
parameters['t_end'] = parameters['t_start'] + parameters['n_timepoints'] * parameters['time_step']

# training parameters
parameters['batch_size'] = 128 # number of training samples (electrograms-activation_maps pairs) are processed together in one pass during training
parameters['learning_rate'] = 1e-4 # too small or too big are both bad
parameters['epochs'] = 100 # maximum epochs (training may stop earlier with early stopping)
parameters['early_stopping_patience'] = 6 # stop training if no improvement for this many epochs

# mode settings
train_predict_flag = 0 # 1: will train the model; 0: only do prediction with the pre-trained model
continue_training = 0 # 0: train from scratch; 1: load best_unet_model.pth and continue training
testing_data_flag = 1 # 0: simulation data; 1: clinical data
data_type = '1focal' # '1focal' or '2focal'

parameters['data_folder_simulation'] = Path('/home/j/Desktop/hdd/share_folder/simulation_results')
parameters['data_folder_patient'] = Path('/home/j/Desktop/hdd/share_folder/patient_data')
parameters['result_folder'] = script_dir / 'result'
parameters['result_folder'].mkdir(exist_ok=True)

#%%
file_names_train, file_names_validation, file_names_test = modules.utility.categorize_files_into_train_validation_test(parameters['data_folder_simulation'])

parameters['file_names_train'] = file_names_train
parameters['file_names_validation'] = file_names_validation
parameters['file_names_test'] = file_names_test

print(f'n_train: {len(file_names_train)}, n_validation: {len(file_names_validation)}, n_test: {len(file_names_test)}')

#%%
# create the U-Net model
try:
   import MinkowskiEngine as ME # https://nvidia.github.io/MinkowskiEngine/overview.html
except ImportError:
   print('MinkowskiEngine is not installed.')
   pass

parameters['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

   # torchview does not support MinkowskiEngine SparseTensor
   # use print to show model architecture instead
   print(parameters['model'])

#%%
# train the model
if train_predict_flag == 1:
   print('train model')

   if continue_training == 1: # load pre-trained model to continue training
      model_path = parameters['result_folder'] / 'best_unet_model.pth'
      print(f'loading pre-trained model from {model_path}')
      parameters['model'].load_state_dict(torch.load(model_path, map_location=parameters['device']))

   # train the model
   train_loss_history, val_loss_history = modules.train_predict.train_model(parameters)

# plot loss history
modules.result_analysis.plot_loss_history(parameters['result_folder'])

#%%
def normalize_to_unit_interval(values):
    min_value = np.min(values)
    max_value = np.max(values)
    range_value = max_value - min_value

    if range_value == 0:
        return np.zeros_like(values, dtype=np.float32)
    
    return ((values - min_value) / range_value).astype(np.float32)

# predict using the trained model
if train_predict_flag == 0:
   # predict with test data
   print('model prediction')

   parameters['model'].load_state_dict(torch.load(parameters['result_folder'] / 'best_unet_model.pth', map_location=parameters['device'])) # load the best model

   if testing_data_flag == 0: # 0: simulation data; 1: clinical data
      predicted_data, truth_data = modules.train_predict.predict_simulation(parameters)
      # convert all elements to numpy arrays if they are tensors
      predicted_data = [x.numpy() if hasattr(x, 'numpy') else x for x in predicted_data]
      truth_data = [x.numpy() if hasattr(x, 'numpy') else x for x in truth_data]
   elif testing_data_flag == 1:
      data_folder_patient = parameters['data_folder_patient']
      file_names = {}
      file_names[0] = '101_1-LA FAM1_processed_map_refined.npz'
      file_names[1] = '102_1-lagood_processed_map_refined.npz'

      parameters['model'].eval()

      n_test_samples = 2
      n_test_batches = (n_test_samples + parameters['batch_size'] - 1) // parameters['batch_size']

      all_predictions = []
      all_truths = []
      with torch.no_grad():
         for batch_idx in range(n_test_batches):
            print(f'  Prediction batch {batch_idx+1}/{n_test_batches}')

            start_idx = batch_idx * parameters['batch_size']
            end_idx = min((batch_idx + 1) * parameters['batch_size'], n_test_samples)

            n_node = parameters['node'].shape[0]
            x_temp = []
            nodes_list = []
            for i in range(start_idx, end_idx):
               name_prefix = file_names[i].split("_processed_map_refined.npz")[0]

               # load patient data to grab the electrode voxel ids
               data = np.load(data_folder_patient / file_names[i], allow_pickle=True)
               map_data = {k: data[k] for k in data.files}

               # find the good electrode nodes that have good signals
               voxel3mm_id_of_electrode = map_data['voxel3mm_id_of_electrode']
               act = map_data['activation_uni']
               good_id = [i for i, x in enumerate(act) if x != 0]

               good_e_id = voxel3mm_id_of_electrode[good_id]
               n_nodes = map_data['voxel3mm_1mm_spacing'].shape[0]
               non_e_id = np.setdiff1d(np.arange(n_nodes), good_e_id)

               voxel3mm_1mm_spacing = map_data['voxel3mm_1mm_spacing']
               voxel3mm_1mm_spacing = voxel3mm_1mm_spacing - np.round(voxel3mm_1mm_spacing.mean(axis=0)).astype(int)
               node = voxel3mm_1mm_spacing # shape (n_nodes, 3)

               b = i - start_idx
               n_nodes = node.shape[0]
               batch_indices = torch.full((n_nodes, 1), b, dtype=torch.int32)
               sample_nodes = torch.cat([batch_indices, torch.from_numpy(node).int()], dim=1) # convert xyz to integers. shape (n_nodes, 4)
               nodes_list.append(sample_nodes)

               # load input data
               electrogram_unipolar = map_data['electrogram_unipolar'].T[parameters['t_start']:parameters['t_end']:parameters['time_step'], :] # shape (t, n_node)
               electrogram_unipolar = normalize_to_unit_interval(electrogram_unipolar)

               x = np.zeros((1000, n_node)) # assign all nodes zero signal
               e_id = map_data['voxel3mm_id_of_electrode']
               x[:, e_id] = electrogram_unipolar[2000-250:2000+250, :] # assign electrode nodes the electrogram signal

               # add a binary row to indicate non-electrode nodes as a mask
               new_row = np.ones((1, x.shape[1]), dtype=np.float32)
               new_row[0, non_e_id] = 0
               x = np.concatenate((x, new_row), axis=0)
               
               x_temp.append(x)

            nodes_batch = torch.cat(nodes_list, dim=0).to(parameters['device'])  # (batch * n_nodes, 4)

            # build feats_batch: (batch * n_nodes, t) — handles variable n_nodes per sample
            feats_list = [torch.from_numpy(x).float().T for x in x_temp]  # each: (n_nodes, t)
            feats_batch = torch.cat(feats_list, dim=0).to(parameters['device'])

            # create MinkowskiEngine sparse tensor
            neural_network_input = ME.SparseTensor(features=feats_batch, coordinates=nodes_batch, device=parameters['device'])

            # # stack into tensors
            # input_data = torch.from_numpy(np.stack(x_temp, axis=0)) # shape (batch, t, n_node)
            # # output_data = torch.from_numpy(np.stack(y_temp, axis=0)) # shape (batch, 2, n_node)

            # # grab time slices
            # input_data = input_data[:, parameters['t_start']:parameters['t_end']:parameters['time_step'], :]

            # input_data = input_data.float().to(parameters['device']) # ensure float32
            # # output_data = output_data.float().to(parameters['device']) # ensure float32

            # device = parameters['device']
            # node = parameters['node']
            # # create nodes_batch for MinkowskiEngine: shape (N_total, 4) where each row is [batch_idx, x, y, z]
            # # node has shape (n_nodes, 3)
            # nodes_list = []
            # current_batch_size = input_data.shape[0]
            # for b in range(current_batch_size):
            #    n_nodes = node.shape[0]

            #    batch_indices = torch.full((n_nodes, 1), b, dtype=torch.int32)
            #    sample_nodes = torch.cat([batch_indices, torch.from_numpy(node).int()], dim=1) # convert xyz to integers. shape (n_nodes, 4)
            #    nodes_list.append(sample_nodes)
            # nodes_batch = torch.cat(nodes_list, dim=0).to(device)  # (batch * n_nodes, 4)

            # # reshape input data: (batch, t, nodes) -> (batch * nodes, t)
            # feats_batch = input_data.permute(0, 2, 1).reshape(-1, input_data.shape[1])
            
            # # create MinkowskiEngine sparse tensor
            # neural_network_input = ME.SparseTensor(features=feats_batch, coordinates=nodes_batch, device=device)

            # forward pass
            outputs = parameters['model'](neural_network_input)

            # current_batch_size = input_data.shape[0]

            # # convert to dense tensor: shape (batch, C, X, Y, Z) for 3D
            # # find the minimum coordinate for dense conversion (required if any coordinate is negative)
            # min_coord = torch.IntTensor(np.array(parameters['node']).min(axis=0).flatten())
            
            # # extract predictions at shifted coordinates
            # dense = outputs.dense(min_coordinate=min_coord)
            # prediction_dense = dense[0].cpu()  # shape: (batch, 2, X, Y, Z)
            # n_nodes = parameters['node'].shape[0]
            # shifted_coord = np.array(parameters['node']).astype(int) - min_coord.numpy() # shift node by min_coord for correct indexing
            # prediction = np.zeros((current_batch_size, 2, n_nodes), dtype=np.float32)
            # for b in range(current_batch_size):
            #    for n, (x, y, z) in enumerate(shifted_coord):
            #       prediction[b, :, n] = prediction_dense[b, :, x, y, z]
            # prediction = torch.tensor(prediction)

            # all_predictions.append(prediction)



         # concatenate all batches
         predicted_data = torch.cat(all_predictions, dim=0).numpy()

      # save the prediction results
      np.save(parameters['result_folder'] / f'predictions.npy', predicted_data)

   #%%
   if testing_data_flag == 0: # 0: simulation data; 1: clinical data
      modules.result_analysis.plot_truth_and_predicted_activation_time_map(truth_data, predicted_data, parameters)

print('done')

#%%
