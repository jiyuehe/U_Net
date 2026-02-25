#%%
import os
from pathlib import Path
script_dir = os.path.dirname(os.path.abspath(__file__)) # get the path of the current script
os.chdir(script_dir) # change the working directory
script_dir = Path(script_dir)

# add the workspace root to Python path
import sys
workspace_root = Path().resolve().parent # Path().resolve() returns an absolute path, the full path
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))
import common

import modules as parent_codes

import torch
import numpy as np
import matplotlib.pyplot as plt 
from torchview import draw_graph # for visualizing the neural network model architecture

#%%
# parameters
parameters = {}
parameters['t_start'] = 0
parameters['time_step'] = 1
parameters['n_timepoints'] = 300
parameters['t_end'] = parameters['t_start'] + parameters['n_timepoints'] * parameters['time_step']
parameters['batch_size'] = 32 # number of training samples (electrograms-activation_maps pairs) are processed together in one pass during training
parameters['learning_rate'] = 1e-4 # too small or too big are both bad
parameters['epochs'] = 500 # maximum epochs (training may stop earlier with early stopping)
parameters['early_stopping_patience'] = 10 # stop training if no improvement for this many epochs
parameters['data_flag'] = 1 # 0: action potential; 1: electrogram
parameters['geometry_flag'] = 1 # 0: 2D sheet, 1: patient 3D atrium

train_flag = 0 # 1: will train the model; 0: only do prediction with the pre-trained model
continue_training = 0 # 1: load best_unet_model.pth and continue training; 0: train from scratch

# geometry
if parameters['geometry_flag'] == 0:
   geometry_file_name = script_dir.parent / '0_data' / 'sheet.npy'
   data_folder_name = '2d data, 2 focal 2 location 15ms apart'
   parameters['result_folder'] = script_dir / 'result_2d'
   parameters['grid_height'] = 128 # do not change
   parameters['grid_width'] = 128 # do not change
elif parameters['geometry_flag'] == 1:
   # name_prefix = '6-1-1-1-LA PACING CL 300 FROM CS 13 14'
   # name_prefix = '6-1-1-LA PACING CL 270 FROM CS3 4'
   name_prefix = '6-1-LA PACING CS 11 12 300CL'

   geometry_file_name = script_dir.parent / '0_data' / f'{name_prefix}_processed.npz'
   data_folder_name = 'atrium 2, 2 focal 2 location 15ms apart'
   parameters['result_folder'] = script_dir / 'result'
   parameters['grid_height'] = [] # unused; for code compatibility
   parameters['grid_width'] = [] # unused; for code compatibility
elif parameters['geometry_flag'] == 4:
   geometry_file_name = script_dir.parent / '0_data' / 'hollow_slab.npy'
   data_folder_name = '3d data, 2 focal 2 location 15ms apart'
   parameters['result_folder'] = script_dir / 'result_3d'
   parameters['grid_height'] = [] # unused; for code compatibility
   parameters['grid_width'] = [] # unused; for code compatibility

#%%
if parameters['geometry_flag'] in [1, 4]:
   import MinkowskiEngine as ME # https://nvidia.github.io/MinkowskiEngine/overview.html

#%%
# load geometry
data = np.load(geometry_file_name, allow_pickle=True)
geometry_data = {k: data[k] for k in data.files}

parameters['node'] = geometry_data['voxel'] # shape: (nodes, 3)
n_nodes = parameters['node'].shape[0]

e_id = geometry_data['electrode_node_id']

n_electrode = len(e_id)
coef = n_electrode / n_nodes
print(f'n_node: {n_nodes}, n_electrode: {n_electrode}, percentage: {coef*100:.2f}%')

parameters['e_id'] = e_id
parameters['non_e_id'] = np.setdiff1d(np.arange(n_nodes), parameters['e_id'])

debug_plot = 0
if debug_plot == 1:
   plt.figure()
   ax = plt.axes(projection='3d')
   ax.scatter(parameters['node'][parameters['non_e_id'], 0], parameters['node'][parameters['non_e_id'], 1], parameters['node'][parameters['non_e_id'], 2], c='grey', s=1, edgecolor='none', linewidth=0, alpha=0.3)
   ax.scatter(parameters['node'][parameters['e_id'], 0], parameters['node'][parameters['e_id'], 1], parameters['node'][parameters['e_id'], 2], c='blue', edgecolor='none', linewidth=0)
   plt.axis('off')
   common.set_axes_equal.execute(ax)
   ax.view_init(elev=90, azim=-90)
   plt.tight_layout()
   plt.savefig(parameters['result_folder'] / f'electrode_{coef}.png', dpi=300, bbox_inches="tight", pad_inches=0)
   plt.close()

#%%
# load the index file
if parameters['geometry_flag'] == 0:
   parameters['data_folder'] = Path('/home/j/Desktop/hdd') / data_folder_name
elif parameters['geometry_flag'] in [1, 4]:
   parameters['data_folder'] = Path('/data') / data_folder_name # this is when using the MinkowskiEngine docker container

#%%
# create the U-Net model
parameters['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if parameters['geometry_flag'] == 0:
   parameters['model'] = parent_codes.unet.UNet(parameters['n_timepoints'], 2).to(parameters['device'])
elif parameters['geometry_flag'] in [1, 4]:
   parameters['model'] = parent_codes.unet_minkowski.MinkowskiUNet(in_channels=parameters['n_timepoints'], out_channels=2,D=3).to(parameters['device']) # D is the dimension of the input data

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
   elif parameters['geometry_flag'] in [1, 4]:
      # torchview does not support MinkowskiEngine SparseTensor
      # use print to show model architecture instead
      print(parameters['model'])

# load data file index
# load training data file index
n_files_to_use = -1 # -1: use all files; or specify a number
parameters['s1_train'], parameters['s2_train'] = parent_codes.load_data.file_index(parameters['data_folder'] / 'train', n_files_to_use)

# load validation data file index
n_files_to_use = -1 # -1: use all files; or specify a number
parameters['s1_validation'], parameters['s2_validation'] = parent_codes.load_data.file_index(parameters['data_folder'] / 'validation', n_files_to_use)

# load test data file index
n_files_to_use = 10 # -1: use all files; or specify a number
parameters['s1_test'], parameters['s2_test'] = parent_codes.load_data.file_index(parameters['data_folder'] / 'test', n_files_to_use)

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
   train_loss_history, val_loss_history = parent_codes.train_predict.train_model(parameters)

# plot loss history
parent_codes.result_analysis.plot_loss_history(parameters['result_folder'], parameters['s1_train'])

#%%
if train_flag == 0:
   # predict with test data
   print('model prediction')

   parameters['model'].load_state_dict(torch.load(parameters['result_folder'] / 'best_unet_model.pth', map_location=parameters['device'])) # load the best model

   testing_data_flag = 1 # 0: simulation data; 1: clinical data
   if testing_data_flag == 0:
      predicted_data, truth_data = parent_codes.train_predict.predict(parameters)
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
               electrogram_unipolar = geometry_data['clinical_electrogram_unipolar_woi'].T # shape (t, n_nodes)
               # electrogram_unipolar = simulation_egm
               electrogram_unipolar = (electrogram_unipolar - np.min(electrogram_unipolar)) / (np.max(electrogram_unipolar) - np.min(electrogram_unipolar)) # normalize to 0-1
               
               x = np.zeros((300, n_node)) # assign all nodes zero signal

               ##########
               x[:, e_id] = electrogram_unipolar # assign electrode nodes the electrogram signal
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
         predictions = torch.cat(all_predictions, dim=0).numpy()

   # save the prediction results
   np.save(parameters['result_folder'] / f'predictions_{name_prefix}.npy', predictions)

   #%%
   if testing_data_flag == 0:
      # plot mix rhythm activation time map
      start_idx = 0
      end_idx = len(parameters['s1_test'])

      # plot full mix rhythm data
      sparse_electrode_flag = 0 # 1: use sparse electrode nodes; 0: use all nodes
      parent_codes.result_analysis.plot_mix_rhythm_activation_time_map(sparse_electrode_flag, start_idx, end_idx, parameters)

      # plot sparse electrode nodes mix rhythm data
      sparse_electrode_flag = 1 # 1: use sparse electrode nodes; 0: use all nodes
      parent_codes.result_analysis.plot_mix_rhythm_activation_time_map(sparse_electrode_flag, start_idx, end_idx, parameters)

      # plot truth and predicted activation time map
      parent_codes.result_analysis.plot_truth_and_predicted_activation_time_map(truth_data, predicted_data, parameters)

print('done')

#%%
