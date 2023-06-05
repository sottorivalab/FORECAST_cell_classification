import os
import sys
import pathlib

import sccnn_detection as sccnn
from subpackages import NetworkOptions

exp_dir = os.path.normpath(str(pathlib.Path(sys.argv[1])))
data_dir = os.path.normpath(str(pathlib.Path(sys.argv[2])))
results_dir = os.path.normpath(str(pathlib.Path(sys.argv[3])))

if len(sys.argv) > 4:
    batch_size = int(sys.argv[4])
else:
    batch_size = 90

if len(sys.argv) > 5:
    file_name_pattern = os.path.normpath(str(pathlib.Path(sys.argv[5])))
else:
    file_name_pattern = "*"
    
if len(sys.argv) > 6 and len(sys.argv[6]) > 0:
    tissue_segment_dir = os.path.normpath(str(pathlib.Path(sys.argv[6])))
else:
    tissue_segment_dir = ''

opts = NetworkOptions.NetworkOptions(exp_dir=exp_dir,
                                     num_examples_per_epoch_train=1,
                                     num_examples_per_epoch_valid=1,
                                     image_height=31,
                                     image_width=31,
                                     in_feat_dim=4,
                                     label_height=13,
                                     label_width=13,
                                     in_label_dim=1,
                                     batch_size=batch_size,
                                     data_dir=data_dir,
                                     results_dir=results_dir,
                                     tissue_segment_dir=tissue_segment_dir,
                                     file_name_pattern=file_name_pattern,
                                     pre_process=True,
                                     maxclique_distance=5,
                                     maxclique_threshold=0.35)

opts.results_dir = os.path.join(opts.results_dir, '20180117')
opts.preprocessed_dir = os.path.join(opts.preprocessed_dir, '20180117')

if not os.path.isdir(opts.results_dir):
    os.makedirs(opts.results_dir)
if not os.path.isdir(os.path.join(opts.results_dir, 'h5')):
    os.makedirs(os.path.join(opts.results_dir, 'h5'))
if not os.path.isdir(os.path.join(opts.results_dir, 'annotated_images')):
    os.makedirs(os.path.join(opts.results_dir, 'annotated_images'))
if not os.path.isdir(os.path.join(opts.results_dir, 'csv')):
    os.makedirs(os.path.join(opts.results_dir, 'csv'))
if not os.path.isdir(os.path.join(opts.preprocessed_dir, 'pre_processed')):
    os.makedirs(os.path.join(opts.preprocessed_dir, 'pre_processed'))

Network = sccnn.SCCNN(batch_size=opts.batch_size,
                      image_height=opts.image_height,
                      image_width=opts.image_width,
                      in_feat_dim=opts.in_feat_dim,
                      out_height=opts.label_height,
                      out_width=opts.label_width,
                      out_feat_dim=opts.in_label_dim,
                      radius=10)

Network = Network.generate_output(opts=opts)
