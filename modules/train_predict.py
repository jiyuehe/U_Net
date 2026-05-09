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

import torch
import torch.nn as nn
import numpy as np
import time
from . import utility

def mse_loss(predictions, targets):
    # extract features from sparse tensors
    pred_features = predictions.F
    target_features = targets.F

    loss = nn.functional.mse_loss(pred_features, target_features)

    return loss

def train_model(parameters):
    # assign the loss function
    criterion = mse_loss

    # Adam optimizer with weight decay
    optimizer = torch.optim.Adam(parameters['model'].parameters(), parameters['learning_rate'], weight_decay=1e-3)
    # weight decay is a regularization technique that adds a penalty to the loss function based on the magnitude of the model's weights. It helps prevent overfitting by discouraging the model from having very large weight values. 
    # weight = weight - learning_rate * gradient - learning_rate * weight_decay * weight

    # reduces learning rate by "factor" if no improvement after "patience" epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3)
    
    # calculate number of batches
    n_train_samples = len(parameters['file_names_train'])
    n_validation_samples = len(parameters['file_names_validation'])
    n_train_batches = (n_train_samples + parameters['batch_size'] - 1) // parameters['batch_size']
    n_validation_batches = (n_validation_samples + parameters['batch_size'] - 1) // parameters['batch_size']

    # training loop
    best_loss = float('inf')
    epochs_without_improvement = 0
    train_loss_history = []
    val_loss_history = []
    
    # create loss history file with header
    loss_file = open(parameters['result_folder'] / 'loss_history.txt', 'w')
    loss_file.write('train_loss\tval_loss\n')

    for epoch in range(parameters['epochs']):
        print(f'Epoch {epoch+1}')
        epoch_start_time = time.time()

        # shuffle training indices at the start of each epoch
        perm = np.random.permutation(n_train_samples)
        file_names_train = np.array(parameters['file_names_train'])[perm]

        name_prefixes = [file_name.split('_simulation_results_')[0] for file_name in file_names_train] # extract name prefixes for loading clinical data
        
        # training phase
        # ------------------------------
        parameters['model'].train() # set model to training mode
        train_loss = 0.0
        
        # for each batch in training data
        for batch_idx in range(n_train_batches):
            print(f'  Training batch {batch_idx+1}/{n_train_batches}          ', end='\r') # '\r' to overwrite the same line

            # load batch data
            start_idx = batch_idx * parameters['batch_size']
            end_idx = min((batch_idx + 1) * parameters['batch_size'], n_train_samples)

            neural_network_input, target_sparse = utility.load_input_and_target(start_idx, end_idx, file_names_train, name_prefixes, parameters, data_type='simulation')
            # print(output_data.shape)

            # set gradients to zero
            optimizer.zero_grad() 

            # forward pass: model processes input_data -> predicted activation maps
            outputs = parameters['model'](neural_network_input) # this calls model.forward(input_data)
            # In PyTorch, when you define a model as a subclass of nn.Module, the class implements a special Python method called __call__(). __call__() (defined in nn.Module) -> calls model.forward(). Therefore model(input_data) calls model.forward(input_data).
            
            # calculate loss
            loss = criterion(outputs, target_sparse)
            
            # backward pass: Compute gradients via backpropagation
            loss.backward()
            
            # update model parameters using Adam optimizer
            optimizer.step()
            
            # accumulate training loss
            train_loss += loss.item()
        
        # average training loss over all batches
        train_loss /= n_train_batches
        train_loss_history.append(train_loss)

        # validation phase
        # ------------------------------
        parameters['model'].eval() # set model to evaluation mode
        val_loss = 0.0
        
        # shuffle validation indices at the start of each epoch
        perm = np.random.permutation(n_validation_samples)
        file_names_validation = np.array(parameters['file_names_validation'])[perm]

        name_prefixes = [file_name.split('_simulation_results_')[0] for file_name in file_names_validation] # extract name prefixes for loading clinical data

        with torch.no_grad(): # disables gradient computation
        # why disable gradients during validation?
        # validation does not require gradient calculations since we are not updating model weights.
            # for each batch in validation data
            for batch_idx in range(n_validation_batches):
                print(f'  Validation batch {batch_idx+1}/{n_validation_batches}          ', end='\r') # '\r' to overwrite the same line

                # load batch data
                start_idx = batch_idx * parameters['batch_size']
                end_idx = min((batch_idx + 1) * parameters['batch_size'], n_validation_samples)

                neural_network_input, target_sparse = utility.load_input_and_target(start_idx, end_idx, file_names_validation, name_prefixes, parameters, data_type='simulation')
                
                # forward pass (no gradient tracking)
                outputs = parameters['model'](neural_network_input)

                # calculate loss
                loss = criterion(outputs, target_sparse)
                    
                # accumulate loss
                val_loss += loss.item()

        # explanation: 
        # with torch.no_grad():
        #     # code block
        # is equivalent to 
        # try:
        #     torch.set_grad_enabled(False) # Setup: disable gradients
        #     # code block
        # finally:
        #     torch.set_grad_enabled(True) # Cleanup: re-enable gradients

        # average validation loss over all batches
        val_loss /= n_validation_batches
        val_loss_history.append(val_loss)

        # write current epoch losses to file
        loss_file.write(f'{train_loss}\t{val_loss}\n')
        loss_file.flush() # force to write the data to the file immediately. because by default, file I/O is buffered and may be written after the file is closed.

        # scheduler will automatically adjust learning rate if no improvement
        scheduler.step(val_loss)
        
        # early stopping
        if val_loss < best_loss:
            if val_loss < best_loss - 0.05e-3:
                epochs_without_improvement = 0

            best_loss = val_loss
            torch.save(parameters['model'].state_dict(), parameters['result_folder'] / 'best_unet_model.pth') # save best model

        elif val_loss >= best_loss:
            epochs_without_improvement += 1
            print(f"    no improvement for {epochs_without_improvement} epoch(s) (min val loss: {best_loss*1000:.4f}e-3)")
            
            if epochs_without_improvement >= parameters['early_stopping_patience']:
                print(f"early stopping triggered after {epoch+1} epochs")
                print(f"best validation loss: {best_loss*1000:.4f}e-3")
                break
    
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"training/validation Loss: {train_loss*1000:.4f}e-3 / {val_loss*1000:.4f}e-3")
        print(f"computation time: {epoch_duration:.1f} seconds")

    return train_loss_history, val_loss_history

def predict(parameters, file_names_test, data_type):
    n_out_channel = 1

    parameters['model'].eval()

    n_test_samples = len(file_names_test)
    n_test_batches = (n_test_samples + parameters['batch_size'] - 1) // parameters['batch_size']

    all_predictions = []
    all_truths = []
    with torch.no_grad():
        for batch_idx in range(n_test_batches):
            print(f'  Prediction batch {batch_idx+1}/{n_test_batches}')

            start_idx = batch_idx * parameters['batch_size']
            end_idx = min((batch_idx + 1) * parameters['batch_size'], n_test_samples)

            # load data
            if data_type == 'simulation':
                name_prefixes = [file_name.split('_simulation_results_')[0] for file_name in file_names_test[start_idx:end_idx]] # extract name prefixes for loading clinical data
            elif data_type == 'clinical':
                name_prefixes = [file_name.split('_clinical_data.npz')[0] for file_name in file_names_test[start_idx:end_idx]] # extract name prefixes for loading clinical data

            neural_network_input, target_sparse = utility.load_input_and_target(start_idx, end_idx, file_names_test, name_prefixes, parameters, data_type)

            # forward pass
            outputs = parameters['model'](neural_network_input)

            current_batch_size = end_idx - start_idx

            # get coordinates from the sparse tensor output: shape (batch * n_nodes, 4) — [batch_idx, x, y, z]
            coords = outputs.C.cpu()
            target_coords = target_sparse.C.cpu()

            # compute min coordinate for dense conversion (required if any coordinate is negative)
            min_coord = coords[:, 1:].min(dim=0).values # shape: (3,)

            # extract predictions per sample (n_nodes may differ per sample)
            dense = outputs.dense(min_coordinate=min_coord)
            prediction_dense = dense[0].cpu() # shape: (batch, n_out_channel, X, Y, Z)

            for b in range(current_batch_size):
                # get spatial coords for this sample
                id = coords[:, 0] == b
                sample_coords = (coords[id, 1:] - min_coord).numpy()  # (n_nodes_b, 3)
                n_nodes_b = sample_coords.shape[0]

                # extract predictions at each node coordinate
                pred_b = np.zeros((n_out_channel, n_nodes_b), dtype=np.float32)
                for n, (x, y, z) in enumerate(sample_coords):
                    pred_b[:, n] = prediction_dense[b, :, x, y, z]
                all_predictions.append(pred_b) # shape: (n_out_channel, n_nodes_b)

                # extract truth features for this sample
                id = target_coords[:, 0] == b
                truth_b = target_sparse.F.cpu()[id, :].T  # shape: (n_out_channel, n_nodes_b)
                all_truths.append(truth_b)


        # return as lists since n_nodes may differ per sample
        predictions = all_predictions # list of (n_out_channel, n_nodes_all_samples) arrays
        truths = all_truths # list of (n_out_channel, n_nodes_all_samples) tensors

    return predictions, truths
