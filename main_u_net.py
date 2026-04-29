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

# from torchview import draw_graph # for visualizing the neural network model architecture

#%matplotlib tk 
# make the Matplotlib plot pop up in a window instead of inline in the Jupyter notebook when debugging; change to %matplotlib inline if want to show plots in the notebook

#%%
# mode settings
train_predict_flag = 0 # 1: only do training; 0: only do prediction
testing_data_flag = 1 # 0: simulation data; 1: clinical data
continue_training = 0 # 0: train from scratch; 1: load best_unet_model.pth and continue training

# time samples
parameters = {}
parameters['t_start'] = 0
parameters['time_step'] = 1
parameters['n_timepoints'] = 500
parameters['t_end'] = parameters['t_start'] + parameters['n_timepoints'] * parameters['time_step']

# training parameters
parameters['batch_size'] = 128 # number of training samples (electrograms-activation_maps pairs of 1 simulation on 1 geometry) processed together in one pass during training
parameters['learning_rate'] = 1e-4 # too small or too big are both bad
parameters['epochs'] = 100 # maximum epochs (training may stop earlier with early stopping)
parameters['early_stopping_patience'] = 6 # stop training if no improvement for this many epochs

parameters['data_folder_simulation'] = Path('/home/j/Desktop/hdd/share_folder/simulation_results')
parameters['data_folder_patient'] = Path('/home/j/Desktop/hdd/share_folder/patient_data')
parameters['result_folder'] = script_dir / 'result'
parameters['result_folder'].mkdir(exist_ok=True)

#%%
# categorize the data files into training, validation, and test sets
file_names_train, file_names_validation, file_names_test = modules.utility.categorize_files_into_train_validation_test(parameters['data_folder_simulation'])

parameters['file_names_train'] = file_names_train
parameters['file_names_validation'] = file_names_validation
parameters['file_names_test'] = file_names_test

print(f'n_train: {len(file_names_train)}, n_validation: {len(file_names_validation)}, n_test: {len(file_names_test)}')

#%%
# create the U-Net model
parameters['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
   import MinkowskiEngine as ME # https://nvidia.github.io/MinkowskiEngine/overview.html

   parameters['model'] = modules.unet_minkowski.MinkowskiUNet(in_channels=parameters['n_timepoints']+1, out_channels=1,D=3).to(parameters['device']) 
   # D is the dimension of the input data
   # in_channels is n_timepoints channels for the electrogram, plus 1 channel as an indicator for electrode nodes (1 for electrode nodes, 0 for non-electrode nodes)
   # out_channels is 1 for the predicted activation time map
except ImportError:
   print('MinkowskiEngine is not installed.')

debug_flag = 0
if debug_flag == 1:
   print(f'Model created with {sum(p.numel() for p in parameters["model"].parameters())} parameters')

   # torchview does not support MinkowskiEngine SparseTensor
   # use print to show model architecture
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

# predict using the trained model
if train_predict_flag == 0:
   print('model prediction')

   parameters['model'].load_state_dict(torch.load(parameters['result_folder'] / 'best_unet_model.pth', map_location=parameters['device'])) # load the best model

   if testing_data_flag == 0: # 0: simulation data; 1: clinical data
      # uniformly sample N files
      N = 3
      file_names_test = np.array(parameters['file_names_test'])
      if len(file_names_test) > N:
         indices = np.linspace(0, len(file_names_test) - 1, N, dtype=int)
         file_names_test = file_names_test[indices]

      predicted_data, truth_data, file_names_test = modules.train_predict.predict(parameters, file_names_test, data_type='simulation')
   elif testing_data_flag == 1:
      file_names_test = {}
      file_names_test[0] = '101_1-LA FAM1_processed_map_refined.npz'
      file_names_test[1] = '102_1-lagood_processed_map_refined.npz'

      predicted_data, truth_data, file_names_test = modules.train_predict.predict(parameters, file_names_test, data_type='clinical')

   # convert all elements to numpy arrays if they are tensors
   predicted_data = [x.numpy() if hasattr(x, 'numpy') else x for x in predicted_data]
   truth_data = [x.numpy() if hasattr(x, 'numpy') else x for x in truth_data]

   # plot the predicted and true activation time maps
   if testing_data_flag == 0: # 0: simulation data; 1: clinical data
      modules.result_analysis.plot_truth_and_predicted_activation_time_map(truth_data, predicted_data, file_names_test, parameters, data_type='simulation')
   elif testing_data_flag == 1:
      modules.result_analysis.plot_truth_and_predicted_activation_time_map(truth_data, predicted_data, file_names_test, parameters, data_type='clinical')

print('done')

#%%
