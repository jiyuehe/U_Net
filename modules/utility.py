import numpy as np

def categorize_files_into_train_validation_test(data_folder_simulation):
    train_validation_test_file_names = 'train_validation_test_file_names.txt'

    if not (data_folder_simulation / train_validation_test_file_names).exists(): # if file not exist
        # grab file names of the simulation results
        simulation_files = list(data_folder_simulation.glob('*.npz'))

        n_samples = len(simulation_files)

        # randomly split into training, validation, and testing
        perm = np.random.permutation(n_samples)

        n_train = int(0.8 * n_samples)
        n_val = int(0.1 * n_samples)

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

