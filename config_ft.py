import tensorflow as tf
import os

flags = tf.flags

# General network setup
flags.DEFINE_string('task', 'train', 'train or test')

# Training logs
flags.DEFINE_integer('max_epoch', 100, 'maximum number of training epochs')

# Hyper-parameters
# For training
flags.DEFINE_integer('batchSize', 35, 'training batch size')
flags.DEFINE_integer('val_batch_size', 35, 'validation batch size')
flags.DEFINE_integer('test_batch_size', 35, 'test batch size')
flags.DEFINE_float('init_lr', 1e-5, 'Initial learning rate')

# Data
flags.DEFINE_string('data_path', 'C:\\Users\\giova\\Downloads\\Dataset_NoLabels\\food101_resized', 'Data path')
flags.DEFINE_integer('img_dim', 225, 'Size of the crop extracted from each input image')
flags.DEFINE_integer('numChannels', 3, 'Input channel size')
flags.DEFINE_integer('num_clas', 101, 'Number of classification classes')

# Directories and settings
flags.DEFINE_string('resources', os.path.join('resources', 'h5_files'), 'Path to h5 files folder')
flags.DEFINE_string('run_name', 'run01', 'Run name')
flags.DEFINE_string('trial_dir', 'ResultsFT', 'Results saving directory')
flags.DEFINE_integer('reload_step', 0, 'Reload step to continue training')
flags.DEFINE_string('weights', 'ResultsJPS/01_run/model_dir/weights.06-4.97.hdf5', 'Path to model weights')


conf_ft = tf.flags.FLAGS