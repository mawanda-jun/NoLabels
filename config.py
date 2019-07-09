import tensorflow as tf
import os

flags = tf.flags

# General network setup
flags.DEFINE_string('mode', 'train', 'train or test')
flags.DEFINE_string('model', 'alexnet', 'matrix_capsule or vector_capsule or alexnet')

# hamming set
flags.DEFINE_integer('hammingSetSize', 100, 'Hamming set size')
flags.DEFINE_string('selectionMethod', 'max', 'max or mean')
flags.DEFINE_string('hammingFileName', 'max_hamming_set_', 'Name of the file to be saved')

# jigsaw
flags.DEFINE_integer('numCrops', 9, 'The number of jigsaw-puzzle crops')
flags.DEFINE_integer('cellSize', 75, 'The dimensions of the jigsaw input')
flags.DEFINE_integer('tileSize', 64, 'The dimensions of the jigsaw input')
flags.DEFINE_integer('colorJitter', 2, 'Number of pixels for color jittering')
flags.DEFINE_integer('cropSize', 225, 'Size of the crop extracted from each input image')
flags.DEFINE_integer('original_dim', 256, 'Size of the input image')

# Training logs
flags.DEFINE_integer('max_epoch', 69, 'maximum number of training epochs')  # with this configuration this is the number of iterations
# to reach in order to solve 69 puzzles per image

# Hyper-parameters
flags.DEFINE_integer('batchSize', 100, 'training batch size')
flags.DEFINE_integer('val_batch_size', 100, 'validation batch size')  # N_val_imgs/(N_train_imgs / batchSize)
flags.DEFINE_float('init_lr', 1e-4, 'Initial learning dropout_rate')
flags.DEFINE_float('lr_min', 1e-6, 'Minimum learning dropout_rate')

# data
flags.DEFINE_string('data_path', 'Dataset/resources/images/ILSVRC_1000', 'Data path')
flags.DEFINE_integer('height', 64, 'Input height size')
flags.DEFINE_integer('width', 64, 'Input width size')
flags.DEFINE_integer('numChannels', 3, 'Input channel size')

# Directories and settings
flags.DEFINE_string('resources', os.path.join('resources', 'hamming_sets'), 'Path to h5 files folder')
flags.DEFINE_string('run_name', 'prova', 'Run name')
flags.DEFINE_string('trial_dir', os.path.join('ResultsJPS'), 'Results saving directory')
flags.DEFINE_string('model_name', 'model', 'Model file name')
flags.DEFINE_integer('reload_step', 0, 'Reload step to continue training')

conf = tf.flags.FLAGS
