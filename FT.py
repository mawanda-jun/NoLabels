import tensorflow as tf
import os
from shutil import copy
from Dataset.data_generator import DataGenerator
from nets.Siamese_ft import SiameseFT
import logging
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
from utils.logger import set_logger

tf.enable_eager_execution()


class FineTuning:
    def __init__(self, conf):
        self.conf = conf
        self.model = self.build()

        print('Generating the dataset...')
        self.dataset = DataGenerator(conf)
        self.dataset.create()

        print('Shuffling the dataset...')
        self.dataset.shuffle(5)

        print('Creating directory...')
        self.log_dir, self.model_dir, self.save_dir = self.set_dirs()

        print('Copy the configuration inside log dir...')
        copy(os.path.join(os.getcwd(), 'nets', 'Siamese_ft.py'), os.path.join(os.getcwd(), self.log_dir))
        copy(os.path.join(os.getcwd(), 'config_ft.py'), os.path.join(os.getcwd(), self.log_dir))

    def build(self):
        model = SiameseFT(self.conf)
        dummy_x = tf.zeros((1, 225, 225, 3))
        model._set_inputs(dummy_x)

        # freezed_layers = ['CONV1', 'CONV2', 'CONV3', 'CONV4', 'CONV5']
        # for layer in model.layers[:13]:
        #     if freezed_layers.__contains__(layer.name):
        #         layer.trainable = False

        loss = tf.keras.losses.CategoricalCrossentropy()

        acc = tf.keras.metrics.CategoricalAccuracy()

        learning_rate = tf.train.exponential_decay(self.conf.init_lr,
                                                   self.conf.reload_step,
                                                   self.conf.batchSize,
                                                   0.97,
                                                   staircase=False)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=[acc])

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
            self.dataset.get_training_set(),
            epochs=self.conf.max_epoch,
            validation_data=self.dataset.get_validation_set(),
            steps_per_epoch=self.conf.batchSize,
            validation_steps=self.conf.val_batch_size,
            verbose=1,
            callbacks=self.setup_callables()
        )
