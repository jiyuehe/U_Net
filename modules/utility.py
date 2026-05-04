import numpy as np
import torch
import sys

try:
    import MinkowskiEngine as ME
except ImportError:
    ME = None


def _materialize_npz(npz_obj):
    """Eagerly read all arrays from an NpzFile into a plain dict."""
    return {k: npz_obj[k] for k in npz_obj.files}


def _load_npz_with_numpy2_pickle_compat(file_path, allow_pickle=True):
    """Load npz and materialize arrays with NumPy 2.x -> 1.x pickle compatibility."""
    try:
        data = np.load(file_path, allow_pickle=allow_pickle)
        return _materialize_npz(data)
    except ModuleNotFoundError as e:
        if "numpy._core" not in str(e):
            raise

        # NumPy 2.x pickles may reference numpy._core.*; old NumPy exposes numpy.core.*
        sys.modules.setdefault('numpy._core', np.core)
        sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
        sys.modules.setdefault('numpy._core._multiarray_umath', np.core._multiarray_umath)

        data = np.load(file_path, allow_pickle=allow_pickle)
        return _materialize_npz(data)

def categorize_files_into_train_validation_test(data_folder_simulation):
    train_validation_test_file_names = 'train_validation_test_file_names.txt'

    if not (data_folder_simulation / train_validation_test_file_names).exists(): # if file not exist
        # grab file names of the simulation results
        simulation_files = list(data_folder_simulation.glob('*.npz'))

        n_samples = len(simulation_files)

        # randomly split into training, validation, and testing
        perm = np.random.permutation(n_samples)

        n_train = int(0.8 * n_samples)
        n_val = int(0.15 * n_samples)

        file_id_train = perm[:n_train]
        file_id_validation = perm[n_train:n_train + n_val]
        file_id_test = perm[n_train + n_val:]

        # write a text file to save the file names of the training, validation and test data 
        with open(data_folder_simulation / train_validation_test_file_names, 'w') as f:
            f.write('[train]\n')
            for idx in file_id_train:
                f.write(f'{simulation_files[idx].name}\n')
            f.write('[validation]\n')
            for idx in file_id_validation:
                f.write(f'{simulation_files[idx].name}\n')
            f.write('[test]\n')
            for idx in file_id_test:
                f.write(f'{simulation_files[idx].name}\n')

    # load the file names of the training, validation and test data from the text file
    with open(data_folder_simulation / train_validation_test_file_names, 'r') as f:
        lines = f.readlines()
        file_names_train = []
        file_names_validation = []
        file_names_test = []
        current_section = None
        for line in lines:
            line = line.strip()
            if line == '[train]':
                current_section = 'train'
            elif line == '[validation]':
                current_section = 'validation'
            elif line == '[test]':
                current_section = 'test'
            elif line and current_section is not None:
                if current_section == 'train':
                    file_names_train.append(line)
                elif current_section == 'validation':
                    file_names_validation.append(line)
                elif current_section == 'test':
                    file_names_test.append(line)

    return file_names_train, file_names_validation, file_names_test

def normalize_to_unit_interval(values):
    min_value = np.nanmin(values)
    max_value = np.nanmax(values)
    range_value = max_value - min_value

    if range_value == 0:
        return np.zeros_like(values, dtype=np.float32)
    
    return ((values - min_value) / range_value).astype(np.float32)

def load_input_and_target(start_idx, end_idx, file_names, parameters, data_type):
    data_folder_simulation = parameters['data_folder_simulation'] 
    data_folder_patient = parameters['data_folder_patient']

    x_temp = []
    y_temp = []
    nodes_list = []
    for i in range(start_idx, end_idx):
        # if data_type == 'simulation':
        #     name_prefix = file_names[i].split("_simulation_results_")[0]
        # elif data_type == 'clinical':
        #     name_prefix = file_names[i].split("_clinical_data.npz")[0]
        
        # load electrode coordinates
        # ------------------------------
        map_data = _load_npz_with_numpy2_pickle_compat(data_folder_patient / file_names[i], allow_pickle=True)

        voxel3mm_1mm_spacing = map_data['voxel3mm_1mm_spacing']
        node = voxel3mm_1mm_spacing - np.round(voxel3mm_1mm_spacing.mean(axis=0)).astype(int) # center the coordinates at the origin and convert to integers. shape (n_nodes, 3)
        n_nodes = node.shape[0]
        voxel3mm_id_of_electrode = map_data['voxel3mm_id_of_electrode']
        
        b = i - start_idx # batch index, 1 batch corresponds to 1 simulation on 1 geometry
        batch_indices = torch.full((n_nodes, 1), b, dtype=torch.int32)
        temp = torch.cat([batch_indices, torch.from_numpy(node).int()], dim=1) # shape (n_nodes, 4) = [batch_indices | x | y | z]
        nodes_list.append(temp)

        # load electrograms
        # ------------------------------
        # find the good electrode nodes that have good signals
        activation_time = map_data['clinical_activation_uni']
        good_id = [i for i, x in enumerate(activation_time) if x != 0]
        good_e_id = voxel3mm_id_of_electrode[good_id]
        non_e_id = np.setdiff1d(np.arange(n_nodes), good_e_id)

        if data_type == 'simulation':
            simulation_results = dict(np.load(data_folder_simulation / file_names[i], allow_pickle=False))
        
            x = simulation_results['electrogram_unipolar'][parameters['t_start']:parameters['t_end']:parameters['time_step'], :] # shape (t, n_node)
            x = normalize_to_unit_interval(x)
            # NOTE: x contains simulated electrograms for every node

            x[:, non_e_id] = 0 # set electrograms of non-electrode nodes to 0. the non-electrode nodes are according to clinical data
        elif data_type == 'clinical':
            egm = map_data['clinical_electrogram_unipolar'] # shape (n_node, t), here t is from 0 to 2500-1
            egm = egm.T # shape (t, n_node)
            egm = egm[2000-250:2000+250, :] # grab electrogram within the time window of interest
            egm = normalize_to_unit_interval(egm)
            # NOTE: egm contains clinical electrograms of only the clinical electrodes

            x = np.zeros((egm.shape[0], n_nodes), dtype=np.float32) # shape (t, n_node)
            x[:, good_e_id] = egm[:, good_id] # assign the clinical electrograms to the good electrode nodes according to clinical data, and set the rest of the nodes to 0

        # add a binary row to indicate non-electrode nodes as a mask
        new_row = np.ones((1, x.shape[1]), dtype=np.float32)
        new_row[0, non_e_id] = 0
        x = np.concatenate((x, new_row), axis=0)
        
        x_temp.append(x)
        
        # load target activation time
        # ------------------------------
        if data_type == 'simulation':
            y = simulation_results['lat_electrode']
        elif data_type == 'clinical':
            y = np.full(n_nodes, np.nan, dtype=np.float32)
            y[good_e_id] = activation_time[good_id] # assign the clinical activation time to the good electrode nodes according to clinical data

        y = normalize_to_unit_interval(y)
        
        y_temp.append(y)

    nodes_batch = torch.cat(nodes_list, dim=0).to(parameters['device'])  # (batch * n_nodes, 4)

    # build feats_batch: (batch * n_nodes, t) — handles variable n_nodes per sample
    feats_list = [torch.from_numpy(x).float().T for x in x_temp]  # each: (n_nodes, t)
    feats_batch = torch.cat(feats_list, dim=0).to(parameters['device'])

    # build targets_batch: (batch * n_nodes, 1)
    targets_list = [torch.from_numpy(y).float().reshape(-1, 1) for y in y_temp]
    targets_batch = torch.cat(targets_list, dim=0).to(parameters['device'])

    # create MinkowskiEngine sparse tensor
    neural_network_input = ME.SparseTensor(features=feats_batch, coordinates=nodes_batch, device=parameters['device'])
    target_sparse = ME.SparseTensor(features=targets_batch, coordinates=nodes_batch, device=parameters['device'])

    return neural_network_input, target_sparse
