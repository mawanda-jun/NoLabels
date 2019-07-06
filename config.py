import tensorflow as tf
import os

# flags = tf.app.flags
flags = tf.flags

# General network setup
flags.DEFINE_string('mode', 'train', 'train or test')
flags.DEFINE_string('model', 'alexnet', 'matrix_capsule or vector_capsule or alexnet')
# flags.DEFINE_string('loss_type', 'margin', 'spread or margin or cross_entropy')
# flags.DEFINE_boolean('add_decoder', False, 'Adds a fully connected decoder and reconstruction loss')
# flags.DEFINE_float('alpha', 0.0005, 'Regularization coefficient to scale down the reconstruction loss')
# flags.DEFINE_boolean('grad_clip', False, 'Adds gradient clipping to get rid of exploding gradient')
# flags.DEFINE_boolean('L2_reg', True, 'Adds L2-regularization to all the network weights')
# flags.DEFINE_float('lmbda', 5e-04, 'L2-regularization coefficient')
# flags.DEFINE_integer('iter', 1, 'Number of EM-routing iterations')
# flags.DEFINE_boolean('fc', False, 'Adds a fully connected layer at the end of each network')

# Matrix Capsule architecture
# flags.DEFINE_boolean('use_bias', True, 'Adds bias to init capsules')
# flags.DEFINE_boolean('use_BN', True, 'Adds BN before conv1 layer')
# flags.DEFINE_boolean('add_coords', True, 'Adds capsule coordinations')
# flags.DEFINE_integer('A', 32, 'A in Figure 1 of the paper')
# flags.DEFINE_integer('B', 8, 'B in Figure 1 of the paper')
# flags.DEFINE_integer('C', 16, 'C in Figure 1 of the paper')
# flags.DEFINE_integer('D', 16, 'D in Figure 1 of the paper')
# flags.DEFINE_integer('E', 16, 'E in Figure 1 of the paper')

# Vector Capsule architecture
# flags.DEFINE_integer('num_fc_caps', 20, 'Number of caps of the fc layer at the end of each network')
# flags.DEFINE_integer('fc_caps_dim', 16, 'Dimension of caps of the fc layer at the end of each network')
# flags.DEFINE_integer('out_caps_dim', 16, 'Dimension of the output capsules')
# flags.DEFINE_integer('h1', 512, 'Number of hidden units of the first FC layer of the reconstruction network')
# flags.DEFINE_integer('h2', 1024, 'Number of hidden units of the second FC layer of the reconstruction network')
# For margin loss
# flags.DEFINE_float('m_plus', 0.9, 'm+ parameter')
# flags.DEFINE_float('m_minus', 0.1, 'm- parameter')
# flags.DEFINE_float('lambda_val', 0.5, 'Down-weighting parameter for the absent class')

# Vector Capsule architecture
# flags.DEFINE_integer('prim_caps_dim', 128, 'Dimension of the primary capsules')

# hamming set
flags.DEFINE_boolean('generateHammingSet', False, 'Generate a new HammingSet')
flags.DEFINE_integer('hammingSetSize', 5, 'Hamming set size')
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
flags.DEFINE_integer('max_epoch', 26000, 'maximum number of training epochs')  # with this configuration this is the number of iterations
# to reach in order to solve 69 puzzles per image
flags.DEFINE_integer('SUMMARY_FREQ', 100, 'Number of step to save summary')
flags.DEFINE_integer('VAL_FREQ', 1, 'Number of step to evaluate the network on Validation data')

# Hyper-parameters
flags.DEFINE_integer('batchSize', 256, 'training batch size')
flags.DEFINE_integer('val_batch_size', 90, 'validation batch size')  # N_val_imgs/(N_train_imgs / batchSize)
flags.DEFINE_float('init_lr', 1e-4, 'Initial learning dropout_rate')
flags.DEFINE_float('lr_min', 1e-7, 'Minimum learning dropout_rate')
flags.DEFINE_float('dropout_rate', 0.5, 'Rate (== 1-keep_prob) for dropout layer')

# data
flags.DEFINE_integer('N_train_imgs', 70000, 'Total number of training examples')
flags.DEFINE_integer('N_val_imgs', 25000, 'Total number of validation examples')
flags.DEFINE_integer('N_test_imgs', 5000, 'Total number of test examples')
flags.DEFINE_string('data_path', 'Dataset/resources/h5_files/ILSVRC_1e5.h5', 'Data path')
flags.DEFINE_boolean('data_augment', True, 'Adds augmentation to data')
flags.DEFINE_integer('max_angle', 40, 'Maximum rotation angle along each axis; when applying augmentation')
flags.DEFINE_integer('height', 64, 'Input height size')
flags.DEFINE_integer('width', 64, 'Input width size')
flags.DEFINE_integer('depth', 32, 'Input depth size')
flags.DEFINE_integer('numChannels', 3, 'Input channel size')

# Directories and settings
flags.DEFINE_string('resources', os.path.join('resources', 'h5_files'), 'Path to h5 files folder')
flags.DEFINE_string('run_name', 'run01', 'Run name')
flags.DEFINE_string('logdir', './Results/log_dir/', 'Logs directory')
flags.DEFINE_string('modeldir', './Results/model_dir/', 'Saved models directory')
flags.DEFINE_string('savedir', './Results/result/', 'Results saving directory')
flags.DEFINE_string('model_name', 'model', 'Model file name')
flags.DEFINE_integer('reload_step', 0, 'Reload step to continue training')

# args = tf.app.flags.FLAGS
conf = tf.flags.FLAGS
