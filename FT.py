import tensorflow as tf
import os
from shutil import copy
from Dataset.data_generator import DataGenerator
from nets.Siamese_ft import SiameseFT
import logging
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
from utils.logger import set_logger
import numpy as np
import h5py

tf.enable_eager_execution()


class FineTuning:
    def __init__(self, conf):
        self.conf = conf
        self.model = self.build()

        print('Generating the dataset...')
        self.dataset = DataGenerator(conf)

        self.train_set = self.dataset.get_training_set()
        self.val_set = self.dataset.get_validation_set()
        self.test_set = self.dataset.get_test_set()

        print('Creating directory...')
        self.log_dir, self.model_dir, self.save_dir = self.set_dirs()

        print('Copy the configuration inside log dir...')
        copy(os.path.join(os.getcwd(), 'nets', 'Siamese_ft.py'), os.path.join(os.getcwd(), self.log_dir))
        copy(os.path.join(os.getcwd(), 'config_ft.py'), os.path.join(os.getcwd(), self.log_dir))

    def load_weights(self, model):
        weights = h5py.File(self.conf.weights)
        items = weights.items()
        conv1 = [
            np.array(weights['CONV1/CONV1/kernel:0']),
            np.array(weights['CONV1/CONV1/bias:0'])
        ]

        batch_norm_1 = [
            np.array(weights['batch_norm_1/batch_norm_1/gamma:0']),
            np.array(weights['batch_norm_1/batch_norm_1/beta:0']),
            np.array(weights['batch_norm_1/batch_norm_1/moving_mean:0']),
            np.array(weights['batch_norm_1/batch_norm_1/moving_variance:0'])
        ]

        conv2 = [
            np.array(weights['CONV2/CONV2/kernel:0']),
            np.array(weights['CONV2/CONV2/bias:0']),
        ]

        batch_norm_2 = [
            np.array(weights['batch_norm_2/batch_norm_2/gamma:0']),
            np.array(weights['batch_norm_2/batch_norm_2/beta:0']),
            np.array(weights['batch_norm_2/batch_norm_2/moving_mean:0']),
            np.array(weights['batch_norm_2/batch_norm_2/moving_variance:0'])
        ]

        conv3 = [
            np.array(weights['CONV3/CONV3/kernel:0']),
            np.array(weights['CONV3/CONV3/bias:0'])
        ]

        conv4 = [
            np.array(weights['CONV4/CONV4/kernel:0']),
            np.array(weights['CONV4/CONV4/bias:0']),
        ]

        conv5 = [
            np.array(weights['CONV5/CONV5/kernel:0']),
            np.array(weights['CONV5/CONV5/bias:0']),
        ]

        batch_norm_3 = [
            np.array(weights['batch_norm_3/batch_norm_3/gamma:0']),
            np.array(weights['batch_norm_3/batch_norm_3/beta:0']),
            np.array(weights['batch_norm_3/batch_norm_3/moving_mean:0']),
            np.array(weights['batch_norm_3/batch_norm_3/moving_variance:0'])
        ]

        model.layers[0].set_weights(conv1)
        model.layers[1].set_weights(batch_norm_1)
        # model.layers[2] == maxpool
        model.layers[3].set_weights(conv2)
        model.layers[4].set_weights(batch_norm_2)
        # model.layers[5] == maxpool
        model.layers[6].set_weights(conv3)
        model.layers[7].set_weights(conv4)
        model.layers[8].set_weights(conv5)
        model.layers[9].set_weights(batch_norm_3)
        # model.layers[10] == flatten
        # model.layers[11] == dense
        # model.layers[12] == dense

    def build(self):
        model = SiameseFT(self.conf)
        inputs = tf.keras.Input(shape=(self.conf.img_dim, self.conf.img_dim, self.conf.numChannels))
        model.build(inputs.shape)
        # dummy_x = tf.zeros((1, 225, 225, 3))
        # model._set_inputs(dummy_x)

        # freezed_layers = ['CONV1', 'CONV2', 'CONV3', 'CONV4', 'CONV5']
        # for layer in model.layers[:13]:
        #     if freezed_layers.__contains__(layer.name):
        #         layer.trainable = False

        loss = tf.keras.losses.SparseCategoricalCrossentropy()

        acc = tf.keras.metrics.SparseCategoricalAccuracy()

        learning_rate = tf.train.exponential_decay(self.conf.init_lr,
                                                   self.conf.reload_step,
                                                   self.conf.batchSize,
                                                   0.97,
                                                   staircase=False)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=[acc])

        self.load_weights(model)
        # model.load_weights(self.conf.weights, True)
        return model

    def set_dirs(self):
        # set dirs per run
        model_dir = os.path.join(os.getcwd(), self.conf.trial_dir, self.conf.run_name, 'model_dir')
        log_dir = os.path.join(os.getcwd(), self.conf.trial_dir, self.conf.run_name, 'log_dir')
        save_dir = os.path.join(os.getcwd(), self.conf.trial_dir, self.conf.run_name, 'save_dir')
        # set experiment path
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        return log_dir, model_dir, save_dir

    def reload_weights(self):
        list_files = os.listdir(self.model_dir)
        if len(list_files) > 0:
            restore_filename = list_files[-1]
            restore_path = os.path.join(self.model_dir, restore_filename)
            if os.path.isfile(restore_path):
                logging.info("Restoring weights in file {}...".format(restore_filename))
                self.model.load_weights(restore_path)
        else:
            raise FileNotFoundError("Weights are not present in folder. Please set reload_step to 0 "
                                    "or double check restore folder")

    def setup_callables(self):
        # Setup callback to save best weights after each epoch
        checkpointer = ModelCheckpoint(filepath=os.path.join(self.model_dir,
                                                             'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
                                       verbose=0,
                                       save_best_only=True,
                                       save_weights_only=True,
                                       monitor='val_loss',
                                       mode='min')
        # setup callback to register training history
        csv_logger = CSVLogger(os.path.join(self.log_dir, 'log.csv'), append=True, separator=';')

        # setup logger to catch warnings and info messages
        set_logger(os.path.join(self.log_dir, 'train_val.log'))

        # setup callback to retrieve tensorboard info
        tensorboard = TensorBoard(log_dir=self.log_dir,
                                  write_graph=True,
                                  histogram_freq=0,
                                  write_grads=True,
                                  write_images=False,
                                  update_freq=self.conf.batchSize * 10)

        return [checkpointer, csv_logger, tensorboard]

    def train(self):
        self.model.summary()

        self.model.fit(
            self.train_set,
            epochs=self.conf.max_epoch,
            validation_data=self.val_set,
            steps_per_epoch=self.dataset.train_size // self.conf.batchSize,
            validation_steps=self.dataset.val_size // self.conf.batchSize,
            callbacks=self.setup_callables()
        )
