import os
from shutil import copyfile
import pickle
import scipy.io as sio
import sccnn_detection as sccnn
from subpackages import NetworkOptions

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

exp_dir = os.path.normpath(str(pathlib.Path(sys.argv[1])))
#network_param_path = os.path.normpath(str(pathlib.Path(sys.argv[2])))
data_dir = os.path.normpath(str(pathlib.Path(sys.argv[2])))
train_data_filename = os.path.normpath(str(pathlib.Path(sys.argv[3])))
valid_data_filename = os.path.normpath(str(pathlib.Path(sys.argv[4])))

opts = NetworkOptions.NetworkOptions(exp_dir=exp_dir,
                                     num_examples_per_epoch_train=1,
                                     num_examples_per_epoch_valid=1,
                                     image_height=31,
                                     image_width=31,
                                     in_feat_dim=4,
                                     label_height=13,
                                     label_width=13,
                                     in_label_dim=1,
                                     batch_size=250,
                                     num_of_epoch=1000,
                                     data_dir=data_dir,
                                     train_data_filename=train_data_filename,
                                     valid_data_filename=valid_data_filename,
                                     current_epoch_num=0)

if not os.path.isdir(opts.exp_dir):
    os.makedirs(opts.exp_dir)
    os.makedirs(opts.checkpoint_dir)
    os.makedirs(opts.log_train_dir)
    os.makedirs(os.path.join(opts.exp_dir, 'subpackges'))

copyfile('Run_training_main.py', os.path.join(opts.exp_dir, 'Run_training_main.py'))
copyfile('sccnn_detection.py', os.path.join(opts.exp_dir, 'sccnn_detection.py'))
files = os.listdir(os.path.join(os.getcwd(), 'subpackages'))
for file in files:
    if file.endswith('.py'):
        copyfile(os.path.join(os.getcwd(), 'subpackages', file),
                 os.path.join(opts.exp_dir, 'subpackges', file))

mat = {'opts': opts}
sio.savemat(os.path.join(opts.exp_dir, 'opts.mat'), mat)
pickle.dump(opts, open(os.path.join(opts.exp_dir, 'opts.p'), 'wb'))


Network = sccnn.SCCNN(batch_size=opts.batch_size,
                      image_height=opts.image_height,
                      image_width=opts.image_width,
                      in_feat_dim=opts.in_feat_dim,
                      out_height=opts.label_height,
                      out_width=opts.label_width,
                      out_feat_dim=opts.in_label_dim,
                      radius=opts.maxclique_distance)
Network = Network.run_training(opts=opts)
