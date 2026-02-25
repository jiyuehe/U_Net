import torch
import torch.nn as nn
import numpy as np
import modules as parent_codes
import time

try:
    import MinkowskiEngine as ME
except ImportError:
    ME = None

def mse_loss(predictions, targets, geometry_flag):
    if geometry_flag == 0: # 0: 2D sheet
        pred_features = predictions
        target_features = targets
    elif geometry_flag in [1, 4]: # 1: patient 3D atrium, 4: hollow 3D cube
        # extract features from sparse tensors
        pred_features = predictions.F
        target_features = targets.F
    
    # mask = ~torch.isnan(target_features)
    # if mask.sum() == 0: # all NaN
    #     return torch.tensor(0.0, device=pred_features.device)
    # pred_features = pred_features[mask]
    # target_features = target_features[mask]

    loss = nn.functional.mse_loss(pred_features, target_features)

    return loss

def garther_Minkowski_input(input_data, output_data, node, device):
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
    
    # reshape output data: (batch, 2, nodes) -> (batch * nodes, 2)
    targets_batch = output_data.permute(0, 2, 1).reshape(-1, output_data.shape[1])

    # create MinkowskiEngine sparse tensor
    neural_network_input = ME.SparseTensor(features=feats_batch, coordinates=nodes_batch, device=device)
    target_sparse = ME.SparseTensor(features=targets_batch, coordinates=nodes_batch, device=device)

    return neural_network_input, target_sparse

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
    n_train_samples = len(parameters['s1_train'])
    n_validation_samples = len(parameters['s1_validation'])
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
        s1_train_shuffled = parameters['s1_train'][perm]
        s2_train_shuffled = parameters['s2_train'][perm]
        
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
            input_data, output_data = parent_codes.load_data.input_output_data(start_idx, end_idx, parameters['data_folder'], parameters['data_folder'] / 'train', s1_train_shuffled, s2_train_shuffled, parameters['non_e_id'], parameters)

            if parameters['geometry_flag'] == 0: # 0: 2D sheet
                # reshape input data
                neural_network_input = input_data.reshape(input_data.shape[0], parameters['n_timepoints'], parameters['grid_height'], parameters['grid_width']) # (batch, t, nodes) -> (batch, t, grid_height, grid_width)
            elif parameters['geometry_flag'] in [1, 4]: # 1: patient 3D atrium, 4: hollow 3D cube
                neural_network_input, target_sparse = garther_Minkowski_input(input_data, output_data, parameters['node'], parameters['device'])

            # set gradients to zero
            optimizer.zero_grad() 

            # forward pass: model processes input_data -> predicted activation maps
            outputs = parameters['model'](neural_network_input) # this calls model.forward(input_data)
            # In PyTorch, when you define a model as a subclass of nn.Module, the class implements a special Python method called __call__(). __call__() (defined in nn.Module) -> calls model.forward(). Therefore model(input_data) calls model.forward(input_data).
            
            # calculate loss
            if parameters['geometry_flag'] == 0: # 0: 2D sheet
                N = outputs.shape[0]
                truth = output_data
                outputs = outputs.reshape(N, 2, parameters['grid_height']*parameters['grid_width'])
                loss = criterion(outputs, truth, parameters['geometry_flag'])
            elif parameters['geometry_flag'] in [1, 4]: # 1: patient 3D atrium, 4: hollow 3D cube
                loss = criterion(outputs, target_sparse, parameters['geometry_flag'])
            
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
        
        with torch.no_grad(): # disables gradient computation
        # why disable gradients during validation?
        # validation does not require gradient calculations since we are not updating model weights.
            # for each batch in validation data
            for batch_idx in range(n_validation_batches):
                print(f'  Validation batch {batch_idx+1}/{n_validation_batches}          ', end='\r') # '\r' to overwrite the same line

                # load batch data
                start_idx = batch_idx * parameters['batch_size']
                end_idx = min((batch_idx + 1) * parameters['batch_size'], n_validation_samples)
                input_data, output_data = parent_codes.load_data.input_output_data(start_idx, end_idx, parameters['data_folder'], parameters['data_folder'] / 'validation', parameters['s1_validation'], parameters['s2_validation'], parameters['non_e_id'], parameters)

                if parameters['geometry_flag'] == 0: # 0: 2D sheet
                    neural_network_input = input_data.reshape(input_data.shape[0], parameters['n_timepoints'], parameters['grid_height'], parameters['grid_width']) # (batch, t, nodes) -> (batch, t, grid_height, grid_width)
                elif parameters['geometry_flag'] in [1, 4]: # 1: patient 3D atrium, 4: hollow 3D cube
                    neural_network_input, target_sparse = garther_Minkowski_input(input_data, output_data, parameters['node'], parameters['device'])
                
                # forward pass (no gradient tracking)
                outputs = parameters['model'](neural_network_input)

                # calculate loss
                if parameters['geometry_flag'] == 0: # 0: 2D sheet
                    N = outputs.shape[0]
                    truth = output_data
                    outputs = outputs.reshape(N, 2, parameters['grid_height']*parameters['grid_width'])
                    loss = criterion(outputs, truth, parameters['geometry_flag'])
                elif parameters['geometry_flag'] in [1, 4]: # 1: patient 3D atrium, 4: hollow 3D cube
                    loss = criterion(outputs, target_sparse, parameters['geometry_flag'])
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
            print(f"    No improvement for {epochs_without_improvement} epoch(s) (min val loss: {best_loss*1000:.4f}e-3)")
            
            if epochs_without_improvement >= parameters['early_stopping_patience']:
                print(f"Early stopping triggered after {epoch+1} epochs")
                print(f"Best validation loss: {best_loss*1000:.4f}e-3")
                break
    
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"Training Loss: {train_loss*1000:.4f}e-3, Validation Loss: {val_loss*1000:.4f}e-3")
        print(f"Computation time: {epoch_duration:.2f} seconds")

    return train_loss_history, val_loss_history

def predict(parameters):
    parameters['model'].eval()

    n_test_samples = len(parameters['s1_test'])
    n_test_batches = (n_test_samples + parameters['batch_size'] - 1) // parameters['batch_size']

    all_predictions = []
    all_truths = []
    with torch.no_grad():
        for batch_idx in range(n_test_batches):
            print(f'  Prediction batch {batch_idx+1}/{n_test_batches}')

            start_idx = batch_idx * parameters['batch_size']
            end_idx = min((batch_idx + 1) * parameters['batch_size'], n_test_samples)

            # load data
            input_data, output_data = parent_codes.load_data.input_output_data(start_idx, end_idx, parameters['data_folder'], parameters['data_folder'] / 'test', parameters['s1_test'], parameters['s2_test'], parameters['non_e_id'], parameters)
            if parameters['geometry_flag'] == 0: # 0: 2D sheet
                # reshape input data
                neural_network_input = input_data.reshape(input_data.shape[0], parameters['n_timepoints'], parameters['grid_height'], parameters['grid_width'])
            elif parameters['geometry_flag'] in [1, 4]: # 1: patient 3D atrium, 4: hollow 3D cube
                neural_network_input, _ = garther_Minkowski_input(input_data, output_data, parameters['node'], parameters['device'])

            # forward pass
            outputs = parameters['model'](neural_network_input)

            if parameters['geometry_flag'] == 0: # 0: 2D sheet
                # reshape outputs
                N = outputs.shape[0]
                prediction = outputs.reshape(N, 2, parameters['grid_height'] * parameters['grid_width']).cpu()
                truth = output_data.cpu()
            elif parameters['geometry_flag'] in [1, 4]: # 1: patient 3D atrium, 4: hollow 3D cube
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

                # reshape output data: (batch, 2, nodes) -> (batch * nodes, 2) 
                truth = output_data.permute(0, 2, 1).reshape(-1, output_data.shape[1])
                # reshape truth to (current_batch_size, 2, n_nodes)
                truth = truth.reshape(current_batch_size, n_nodes, 2).permute(0, 2, 1)
                truth = truth.cpu()

            all_predictions.append(prediction)
            all_truths.append(truth)

        # concatenate all batches
        predictions = torch.cat(all_predictions, dim=0).numpy()
        truths = torch.cat(all_truths, dim=0).numpy()

    return predictions, truths
