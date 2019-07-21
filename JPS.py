import h5py
import tensorflow as tf
import numpy as np
import os, glob
from shutil import copy
from utils.logger import set_logger
import logging
from Dataset.crops_generator_JPS import CropsGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping
from nets.Siamese_jps import Siamese

tf.enable_eager_execution()


class JigsawPuzzleSolver:
    def __init__(self, conf):
        self.conf = conf

        with h5py.File(
                os.path.join('Dataset', conf.resources, conf.hammingFileName + str(conf.hammingSetSize) + '.h5'), 'r') as h5f:
            self.hamming_set = np.array(h5f['max_hamming_set'])

        self.data_reader = CropsGenerator(conf, self.hamming_set)

        self.model = self.build()

        # if reload_step is not zero it means we are reading data from already existing folders
        self.log_dir, self.model_dir, self.save_dir = self.set_dirs()

        # copy configuration inside log_dir
        copy(os.path.join(os.getcwd(), 'nets', 'Siamese_jps.py'), os.path.join(os.getcwd(), self.log_dir))
        copy(os.path.join(os.getcwd(), 'config.py'), os.path.join(os.getcwd(), self.log_dir))

        self.trained = False

    def build(self):
        # initialize model_on_input
        model = Siamese(self.conf)

        # make placeholder to retrieve information about input shape
        inputs = tf.keras.Input(shape=(self.conf.tileSize, self.conf.tileSize, self.conf.numChannels, self.conf.numCrops))
        model.build(inputs.shape)

        # print summary
        model.summary()
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
        if self.conf.reload_step < 10:
            init_epoch = '0'+str(self.conf.reload_step)
        else:
            init_epoch = str(self.conf.reload_step)
        restore_filename_reg = 'weights.{}-*.hdf5'.format(init_epoch)
        restore_path_reg = os.path.join(self.model_dir, restore_filename_reg)
        list_files = glob.glob(restore_path_reg)
        assert len(list_files) > 0, 'ERR: No weights file match provided name {}'.format(
            restore_path_reg)

        # Take real filename
        restore_filename = list_files[0].split('/')[-1]
        restore_path = os.path.join(self.model_dir, restore_filename)

        assert os.path.isfile(restore_path), \
            'ERR: Weight file in path {} seems not to be a file'.format(restore_path)
        self.model.load_weights(restore_path)

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

        # setup early stopping to stop training if val_loss is not increasing after 3 epochs
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=1,
            mode='min',
        )

        return [checkpointer, csv_logger, tensorboard, early_stopping]

    def compile_model(self, mode='train'):
        # set optimizer
        if mode == 'train':
            steps_per_epoch = self.data_reader.num_train_batch
        elif mode == 'val':
            steps_per_epoch = self.data_reader.num_val_batch
        elif mode == 'test':
            steps_per_epoch = self.data_reader.num_test_batch
        else:
            raise ValueError("Bad mode choosen. Please choose between train, val or test")

        learning_rate = tf.compat.v1.train.exponential_decay(self.conf.init_lr,
                                                   self.conf.reload_step,
                                                   steps_per_epoch,
                                                   self.conf.decay_rate,
                                                   staircase=False)
        if self.conf.optimizer == 'SGD':
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=learning_rate,
                momentum=self.conf.sgd_momentum,
                nesterov=True,
                name='N_SGD_momentum'
            )
        elif self.conf.optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, name='adam')
        else:
            raise ValueError("This optimizer has not been implemented yet. Please choose between 'adam' or 'SGD'")

        # compile model_on_input with losses and metrics. Even if there is reloading, learning rate and optimizer state
        # cannot be saved (yet)
        loss = tf.keras.losses.CategoricalCrossentropy()

        acc = tf.keras.metrics.CategoricalAccuracy()
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[acc]
        )

    def train(self):
        # if reload step is > 0 we would like to retrieve pre-trained weights from disk
        if self.conf.reload_step > 0:
            self.reload_weights()

        # set optimizer, loss and accuracy and compile model_on_input
        self.compile_model('train')

        # fit and validate model_on_input
        self.model.fit(
            self.data_reader.generate_train_set(),
            epochs=self.conf.max_epoch,
            validation_data=self.data_reader.generate_val_set(),
            steps_per_epoch=self.data_reader.num_train_batch,
            validation_steps=self.data_reader.num_val_batch,
            verbose=1,
            callbacks=self.setup_callables(),
            initial_epoch=self.conf.reload_step,
        )
        self.trained = True

    def evaluate(self):
        if not self.trained:
            weight_to_be_restored = os.path.join(self.model_dir, self.conf.eval_weight)
            if not os.path.isfile(weight_to_be_restored):
                raise FileNotFoundError('Weight not found. Please double check trial_dir, run_name and eval_weight')
            self.model.load_weights(weight_to_be_restored, by_name=True)
            self.compile_model('test')
        results = self.model.evaluate(
            self.data_reader.generate_test_set(),
            verbose=1,
            steps=self.data_reader.num_test_batch,
        )
        with open(os.path.join(self.log_dir, 'test_{}.csv'.format(self.conf.eval_weight)), 'w') as f:
            f.write("test loss, test acc: {}".format(results))



