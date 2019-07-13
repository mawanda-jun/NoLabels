import h5py
import tensorflow as tf
import numpy as np
import os
from utils.logger import set_logger
import logging
from Dataset.crops_generator_TF import CropsGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
from nets.Siamese_eager import Siamese

tf.enable_eager_execution()


# device = '/cpu:0' if tfe.num_gpus() == 0 else '/gpu:0'
class JigsawPuzzleSolver:
    def __init__(self, conf):
        self.conf = conf

        with h5py.File(
                os.path.join('Dataset', conf.resources, conf.hammingFileName + str(conf.hammingSetSize) + '.h5'), 'r') as h5f:
            self.hamming_set = np.array(h5f['max_hamming_set'])

        self.model = self.build()

        self.data_reader = CropsGenerator(conf, self.hamming_set)

        # if reload_step is not zero it means we are reading data from already existing folders
        self.log_dir, self.model_dir, self.save_dir = self.set_dirs()

    def build(self):
        # initialize model
        model = Siamese(self.conf.hammingSetSize)
        # make placeholder to retrieve information about input shape
        dummy_input = tf.zeros((1, 64, 64, 3, 9))
        model._set_inputs(dummy_input)
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

    def compile_model(self):
        # set optimizer
        steps_per_epoch = self.data_reader.num_train_batch
        learning_rate = tf.train.exponential_decay(self.conf.init_lr,
                                                   self.conf.reload_step,
                                                   steps_per_epoch,
                                                   0.97,
                                                   staircase=False)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # compile model with losses and metrics. Even if there is reloading, learning rate and optimizer state is
        # not saved (yet)
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

        # set optimizer, loss and accuracy and compile model
        self.compile_model()

        # print summary of the network
        self.model.summary()

        # fit and validate model
        self.model.fit(
            self.data_reader.generate(mode='train'),
            epochs=self.conf.max_epoch,
            validation_data=self.data_reader.generate(mode='val'),
            steps_per_epoch=self.data_reader.num_train_batch,
            validation_steps=self.data_reader.num_val_batch,
            verbose=1,
            callbacks=self.setup_callables(),
            initial_epoch=self.conf.reload_step,
        )

    def evaluate(self):
        weight_to_be_restored = os.path.join(self.model_dir, self.conf.eval_weight)
        if os.path.isfile(weight_to_be_restored):
            self.model.load_weights(weight_to_be_restored)
            self.compile_model()
            results = self.model.evaluate(
                self.data_reader.generate('test'),
                verbose=1,
                steps=self.data_reader.num_test_batch
            )
            with open(os.path.join(self.log_dir, 'test_log.csv'), 'w') as f:
                f.write("test loss, test acc: {}".format(results))
        else:
            raise FileNotFoundError('Weight not found. Please double check trial_dir, run_name and eval_weight')


