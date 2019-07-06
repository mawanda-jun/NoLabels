import h5py
import tensorflow as tf
from config import conf
import numpy as np
import os
from utils.logger import set_logger
import logging
from Dataset.crops_generator_TF import CropsGenerator
from tensorflow.contrib.eager.python import tfe
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger
from nets.Siamese_eager import Siamese

tf.enable_eager_execution()
# device = '/cpu:0' if tfe.num_gpus() == 0 else '/gpu:0'


def train():
    if conf.mode not in ['train', 'test', 'predict']:
        print('invalid mode: ', conf.mode)
        print("Please input a mode: train, test, or predict")
    else:
        # with tf.device(device):
        # set logger path
        set_logger(os.path.join(os.getcwd(), 'train_val.log'))

        # open hamming file
        with h5py.File(
                os.path.join('Dataset', conf.resources, conf.hammingFileName + str(conf.hammingSetSize) + '.h5'),
                'r') as h5f:
            HammingSet = np.array(h5f['max_hamming_set'])

        # initialize model
        model = Siamese(conf.hammingSetSize)

        # make placeholder to retrieve information about input shape
        dummy_input = tf.zeros((1, 64, 64, 3, 9))
        model._set_inputs(dummy_input)

        # set experiment path
        os.makedirs(conf.modeldir + conf.run_name, exist_ok=True)
        os.makedirs(conf.logdir + conf.run_name, exist_ok=True)
        os.makedirs(conf.savedir + conf.run_name, exist_ok=True)
        set_logger(os.path.join(os.getcwd(), 'train_val.log'))

        # check if saved weights are present
        experiment_path = conf.modeldir + conf.run_name
        if conf.reload_step > 0:
            list_files = os.listdir(experiment_path)
            if len(list_files) > 0:
                restore_filename = list_files[-1]
                restore_path = os.path.join(experiment_path, restore_filename)
                if os.path.isfile(restore_path):
                    logging.info("Restoring weights in file {}...".format(restore_filename))
                    model.load_weights(restore_path)

        # set optimizer
        steps_per_epoch = conf.N_train_imgs // conf.batchSize
        learning_rate = tf.train.exponential_decay(conf.init_lr,
                                                   conf.reload_step,
                                                   steps_per_epoch,
                                                   0.97,
                                                   staircase=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # compile model with losses and metrics
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy']
        )

        # print summary of the network
        model.summary()

        # Setup callback to save best weights after each epoch
        checkpointer = ModelCheckpoint(filepath=os.path.join(experiment_path,
                                                             'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
                                       verbose=0,
                                       save_best_only=True,
                                       save_weights_only=True,
                                       monitor='val_loss',
                                       mode='min')
        # setup callback to register training history
        csv_logger = CSVLogger('log.csv', append=True, separator=';')

        # create dataset
        data_reader = CropsGenerator(conf, HammingSet)
        train_set = data_reader.generate(mode='train')
        val_set = data_reader.generate(mode='val')

        # fit and validate model
        model.fit(
            train_set,
            epochs=conf.max_epoch,
            validation_data=val_set,
            steps_per_epoch=data_reader.num_train_batch,
            validation_steps=data_reader.num_val_batch,
            verbose=1,
            callbacks=[checkpointer, csv_logger]
        )


if __name__ == '__main__':
    # configure which gpu or cpu to use
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
    train()
