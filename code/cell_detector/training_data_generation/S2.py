import scipy.io as sio
import numpy as np
import h5py
import os
import pathlib
import glob
import sys

main_file_path = os.path.normpath(str(pathlib.Path(sys.argv[1])))
save_path = os.path.normpath(str(pathlib.Path(sys.argv[2])))

if len(sys.argv) >= 6:
    Train_Data_save_name = os.path.normpath(str(pathlib.Path(sys.argv[3])))
    Valid_Data_save_name = os.path.normpath(str(pathlib.Path(sys.argv[4])))
else:
    Train_Data_save_name = 'TrainData.h5'
    Valid_Data_save_name = 'ValidData.h5'

data_size = [31, 31]
label_size = [13, 13]
features = [['h'], ['rgb']]

nFeatures = 0;

for Feature_n in range(0, len(features)):
    if features[Feature_n][0] == 'rgb':
        nFeatures = nFeatures + 3
    elif features[Feature_n][0] == 'lab':
        nFeatures = nFeatures + 3
    elif features[Feature_n][0] == 'h':
        nFeatures = nFeatures + 1
    elif features[Feature_n][0] == 'he':
        nFeatures = nFeatures + 2
    elif features[Feature_n][0] == 'br':
        nFeatures = nFeatures + 1
    elif features[Feature_n][0] == 'grey':
        nFeatures = nFeatures + 1

hf_train_filename = 'TrainData.h5'
hf_valid_filename = 'ValidData.h5'

train_main_file_path = os.path.join(main_file_path, "Training");
valid_main_file_path = os.path.join(main_file_path, "Validation");

#######################################################################################################################
itr_train = 0
hf_train = h5py.File(os.path.join(save_path, hf_train_filename), 'w-')
data_set_train = hf_train.create_dataset("data", (1, data_size[0], data_size[1], nFeatures), maxshape=(None, data_size[0], data_size[1], nFeatures), dtype='float32')
label_set_train = hf_train.create_dataset("labels", (1, label_size[0], label_size[1], 1), maxshape=(None, label_size[0], label_size[1], 1), dtype='float32')
mat_files = glob.glob(os.path.join(train_main_file_path, 'p*.mat'))
nmat_files = glob.glob(os.path.join(train_main_file_path, 'n*.mat'))

number_of_mat_files = len(mat_files)
number_of_nmat_files = len(nmat_files)

for i in range(number_of_mat_files):
    file_path = mat_files[i]
    if i % 1000 == 0:
        print(file_path)
    workspace = sio.loadmat(file_path)
    data = np.array(workspace['data'])
    labels = np.array(workspace['labels'])
    labels = np.expand_dims(labels, axis=2)
    data_set_train.resize(((itr_train + 1), data_size[0], data_size[1], nFeatures))
    label_set_train.resize(((itr_train + 1), label_size[0], label_size[1], 1))
    data_set_train[itr_train, :, :, :] = data
    label_set_train[itr_train, :, :, :] = labels
    itr_train = itr_train + 1
    data_set_train.resize(((itr_train + 1), data_size[0], data_size[1], nFeatures))
    label_set_train.resize(((itr_train + 1), label_size[0], label_size[1], 1))
    data_set_train[itr_train, :, :, :] = np.fliplr(data)
    label_set_train[itr_train, :, :, :] = np.fliplr(labels)
    itr_train = itr_train + 1
    data_set_train.resize(((itr_train + 1), data_size[0], data_size[1], nFeatures))
    label_set_train.resize(((itr_train + 1), label_size[0], label_size[1], 1))
    data_set_train[itr_train, :, :, :] = np.flipud(data)
    label_set_train[itr_train, :, :, :] = np.flipud(labels)
    itr_train = itr_train + 1
    data_set_train.resize(((itr_train + 1), data_size[0], data_size[1], nFeatures))
    label_set_train.resize(((itr_train + 1), label_size[0], label_size[1], 1))
    data_set_train[itr_train, :, :, :] = np.rot90(data, k=2)
    label_set_train[itr_train, :, :, :] = np.rot90(labels, k=2)
    itr_train = itr_train + 1

for i in range(int(number_of_nmat_files)):
    file_path = nmat_files[i]
    if i % 1000 == 0:
        print(file_path)
    workspace = sio.loadmat(file_path)
    data = np.array(workspace['data'])
    labels = np.array(workspace['labels'])
    labels = np.expand_dims(labels, axis=2)

    data_set_train.resize(((itr_train + 1), data_size[0], data_size[1], nFeatures))
    label_set_train.resize(((itr_train + 1), label_size[0], label_size[1], 1))
    data_set_train[itr_train, :, :, :] = data
    label_set_train[itr_train, :, :, :] = labels
    itr_train = itr_train + 1

hf_train.close()


itr_valid = 0
hf_valid = h5py.File(os.path.join(save_path, hf_valid_filename), 'w-')
data_set_valid = hf_valid.create_dataset("data", (1, data_size[0], data_size[1], nFeatures), maxshape=(None, data_size[0], data_size[1], nFeatures), dtype='float32')
label_set_valid = hf_valid.create_dataset("labels", (1, label_size[0], label_size[1], 1), maxshape=(None, label_size[0], label_size[1], 1), dtype='float32')
mat_files = glob.glob(os.path.join(valid_main_file_path, 'mat', '*.mat'))
nmat_files = glob.glob(os.path.join(valid_main_file_path, 'nmat', '*.mat'))
number_of_mat_files = len(mat_files)
number_of_nmat_files = len(nmat_files)

for i in range(number_of_mat_files):
    file_path = mat_files[i]
    if i % 1000 == 0:
        print(file_path)
    workspace = sio.loadmat(file_path)
    data = np.array(workspace['data'])
    labels = np.array(workspace['labels'])
    labels = np.expand_dims(labels, axis=2)

    data_set_valid.resize(((itr_valid + 1), data_size[0], data_size[1], nFeatures))
    label_set_valid.resize(((itr_valid + 1), label_size[0], label_size[1], 1))
    data_set_valid[itr_valid, :, :, :] = data
    label_set_valid[itr_valid, :, :, :] = labels
    itr_valid = itr_valid + 1

for i in range(int(number_of_nmat_files)):
    file_path = nmat_files[i]
    if i % 1000 == 0:
        print(file_path)
    workspace = sio.loadmat(file_path)
    data = np.array(workspace['data'])
    labels = np.array(workspace['labels'])
    labels = np.expand_dims(labels, axis=2)

    data_set_valid.resize(((itr_valid + 1), data_size[0], data_size[1], nFeatures))
    label_set_valid.resize(((itr_valid + 1), label_size[0], label_size[1], 1))
    data_set_valid[itr_valid, :, :, :] = data
    label_set_valid[itr_valid, :, :, :] = labels
    itr_valid = itr_valid + 1

hf_valid.close()
