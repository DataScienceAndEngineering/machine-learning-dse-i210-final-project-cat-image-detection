import os
import shutil
import numpy as np

def split_data(dataset_path, training_path, validation_path, test_path, data_size):
    data = []

    for file in os.scandir(dataset_path):
        if os.path.getsize(file.path) > 0:
            data.append(file.name)

    training_size = int(data_size * 0.8)  # 80% of the data for training
    validation_size = int(data_size * 0.15)  # 15% of the data for validation
    test_size = data_size - training_size - validation_size  # 5% of the data for testing

    shuffled_data = np.random.choice(data, size=data_size, replace=True)

    training_data = shuffled_data[0: training_size]
    validation_data = shuffled_data[training_size: training_size + validation_size]
    test_data = shuffled_data[training_size + validation_size:]

    for file in training_data:
        shutil.copy(os.path.join(dataset_path, file), os.path.join(training_path, file))

    for file in validation_data:
        shutil.copy(os.path.join(dataset_path, file), os.path.join(validation_path, file))

    for file in test_data:
        shutil.copy(os.path.join(dataset_path, file), os.path.join(test_path, file))