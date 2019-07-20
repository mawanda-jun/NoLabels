import tensorflow as tf
import os

flags = tf.flags

# General network setup
flags.DEFINE_string('task', 'train', 'train or test')
flags.DEFINE_string('model_on_input', 'alexnet', 'matrix_capsule or vector_capsule or alexnet')

# Training logs
flags.DEFINE_integer('max_epoch', 10000, 'maximum number of training epochs')
flags.DEFINE_integer('SUMMARY_FREQ', 100, 'Number of step to save summary')
flags.DEFINE_integer('VAL_FREQ', 1000, 'Number of step to evaluate the network on Validation data')

# Hyper-parameters
# For training
flags.DEFINE_integer('batchSize', 5, 'training batch size')
flags.DEFINE_integer('val_batch_size', 64, 'validation batch size')
flags.DEFINE_integer('test_batch_size', 64, 'test batch size')
flags.DEFINE_float('init_lr', 1e-3, 'Initial learning rate')
flags.DEFINE_float('lr_min', 1e-4, 'Minimum learning rate')

# Data
flags.DEFINE_string('data_path', 'Dataset/resources/food_101', 'Data path')
flags.DEFINE_boolean('data_augment', True, 'Adds augmentation to data')
flags.DEFINE_integer('max_angle', 40, 'Maximum rotation angle along each axis; when applying augmentation')
flags.DEFINE_integer('num_tr', 55000, 'Total number of training images')
flags.DEFINE_integer('height', 64, 'Input height size')
flags.DEFINE_integer('width', 64, 'Input width size')
flags.DEFINE_integer('depth', 32, 'Input depth size')
flags.DEFINE_integer('numChannels', 3, 'Input channel size')
flags.DEFINE_integer('num_clas', 100, 'Number of classification classes')
flags.DEFINE_float('train_size', 0.7, 'Size in % of the training dataset')
flags.DEFINE_float('val_size', 0.25, 'Size in % of the validation dataset')
flags.DEFINE_float('test_size', 0.05, 'Size in % of the test dataset')

# Hamming set
flags.DEFINE_boolean('generateHammingSet', False, 'Generate a new HammingSet')
flags.DEFINE_integer('hammingSetSize', 100, 'Hamming set size')
flags.DEFINE_string('selectionMethod', 'max', 'max or mean')
flags.DEFINE_string('hammingFileName', 'max_hamming_set_69.h5', 'Name of the file to be saved')

# Jigsaw
flags.DEFINE_integer('numCrops', 9, 'The number of jigsaw-puzzle crops')
flags.DEFINE_integer('cellSize', 75, 'The dimensions of the jigsaw input')
flags.DEFINE_integer('tileSize', 64, 'The dimensions of the jigsaw input')
flags.DEFINE_integer('colorJitter', 2, 'Number of pixels for color jittering')
flags.DEFINE_integer('cropSize', 225, 'Size of the crop extracted from each input image')

# Directories and settings
flags.DEFINE_string('resources', os.path.join('resources', 'h5_files'), 'Path to h5 files folder')
flags.DEFINE_string('run_name', 'run01', 'Run name')
flags.DEFINE_string('trial_dir', os.path.join('ResultsFT'), 'Results saving directory')
flags.DEFINE_integer('reload_step', 0, 'Reload step to continue training')
flags.DEFINE_string('weights', 'ResultsJPS/run01/model_dir/weights.01-7.31.hdf5', 'Path to model weights')


conf_ft = tf.flags.FLAGS