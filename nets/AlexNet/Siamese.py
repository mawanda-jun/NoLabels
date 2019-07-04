"""
Copyright 2017-2022 Department of Electrical and Computer Engineering
University of Houston, TX/USA
**********************************************************************************
Author:   Aryan Mobiny
Date:     8/1/2017
Comments: ResNet with 50 convolutional layer for classifying the 3D lung nodule images.
This network is almost similar to the one with 50 layer used in the original
paper: "Deep Residual Learning for Image Recognition"
**********************************************************************************
"""
from nets.AlexNet.loss_ops import cross_entropy_loss
from nets.AlexNet.ops import *
from nets.AlexNet.AlexNet import AlexNet
import numpy as np
from Dataset.crops_generator_TF import CropsGenerator
import os
import logging


class Siamese_AlexNet(object):
    def __init__(self, sess, conf, hamming_set):
        self.sess = sess
        self.conf = conf
        _, self.global_step_int = self.check_if_model_exists(conf.reload_step)
        self.HammingSet = hamming_set
        self.input_shape = [None, conf.tileSize, conf.tileSize, conf.numChannels, conf.numCrops]

        self.data_reader = CropsGenerator(self.conf, self.HammingSet)
        self.train_set = self.data_reader.generate(mode='train')
        self.train_iter = self.train_set.make_initializable_iterator()
        self.val_set = self.data_reader.generate(mode='val')
        self.val_iter = self.val_set.make_initializable_iterator()
        self.x, self.y, self.is_train = self.create_placeholders()
        self.dropout_rate = conf.dropout_rate

        self.inference()
        self.configure_network()

    def create_placeholders(self):
        with tf.name_scope('Input'):
            # create sort of placeholder to feed the net with a generic type to feed the net with different datasets.
            self.handle = tf.placeholder(tf.string, shape=[])
            # create iterator to feed the net with values from "get_next"
            # Before train and evaluation self.iter will change and so will the referring dataset
            self.handle_iter = tf.data.Iterator.from_string_handle(self.handle, self.train_set.output_types, self.train_set.output_shapes)
            x, y = self.handle_iter.get_next()
            # placeholder for keep probability in dropout layers
            is_train = tf.placeholder(tf.bool, name='is_train')
        return x, y, is_train

    def inference(self):
        # Build the Network
        with tf.variable_scope('Siamese', reuse=tf.AUTO_REUSE) as scope:
            Siamese_out = []
            # now x is the iterator defined above. We unstack it per numCrops dimension
            x = tf.unstack(self.x, axis=-1)
            for i in range(self.conf.numCrops):
                # stacking different nets together
                Siamese_out.append(AlexNet(x[i], self.dropout_rate, self.is_train))
                # if i < self.conf.numCrops:
                    # Share parameters defined inside <scope>.
                    # Inside AlexNet the name of the layers are defined. Every
                    # iteration of this for loop will use the layers between all nets
                    # scope.reuse_variables()

        net = tf.concat(Siamese_out, axis=1)
        # last layers with which we make inference
        net = fc_layer(net, 4096, 'FC7', is_train=self.is_train, batch_norm=True, activation='softmax')
        net = dropout(net, self.dropout_rate, self.is_train)
        # logits are another name to call the labels, y or whatever
        # here we use linear activation since the loss function calculates the loss with a more efficient softmax activation
        self.logits = fc_layer(net, self.conf.hammingSetSize, 'FC8',
                               is_train=self.is_train, activation='linear')

    def accuracy_func(self):
        with tf.name_scope('Accuracy'):
            correct_prediction = tf.equal(tf.math.argmax(self.logits, 1), tf.math.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.mean_accuracy, self.mean_accuracy_op = tf.metrics.mean(accuracy)

    def loss_func(self):
        with tf.name_scope('Loss'):
            self.total_loss = cross_entropy_loss(self.y, self.logits)
            self.mean_loss, self.mean_loss_op = tf.metrics.mean(self.total_loss)

    def configure_network(self):
        self.loss_func()
        self.accuracy_func()
        with tf.name_scope('Optimizer'):
            with tf.name_scope('Learning_rate_decay'):
                global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(self.global_step_int),
                                              trainable=False)
                steps_per_epoch = self.conf.N_train_imgs // self.conf.batchSize
                learning_rate = tf.train.exponential_decay(self.conf.init_lr,
                                                           global_step,
                                                           steps_per_epoch,
                                                           0.97,
                                                           staircase=True)
                self.learning_rate = tf.maximum(learning_rate, self.conf.lr_min)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.total_loss, global_step=global_step)
        self.sess.run(tf.global_variables_initializer())
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=1000)
        self.train_writer = tf.summary.FileWriter(self.conf.logdir + self.conf.run_name + '/train/', self.sess.graph)
        self.valid_writer = tf.summary.FileWriter(self.conf.logdir + self.conf.run_name + '/valid/')
        self.configure_summary()
        logging.info('Total number of trainable parameters: {}'.
              format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

    def configure_summary(self):
        summary_list = [tf.summary.scalar('Loss/total_loss', self.mean_loss),
                        tf.summary.scalar('Accuracy/accuracy', self.mean_accuracy),
                        tf.summary.scalar('Learning_rate', self.learning_rate),
                        # tf.summary.image('reconstructed', recon_img),
                        ]
        self.merged_summary = tf.summary.merge(summary_list)

    def save_summary(self, summary, step, mode):
        # print('----> Summarizing at step {}'.format(step))
        if mode == 'train':
            self.train_writer.add_summary(summary, step)
        elif mode == 'valid':
            self.valid_writer.add_summary(summary, step)
        self.sess.run(tf.local_variables_initializer())

    def train(self):
        self.sess.run(tf.local_variables_initializer())
        self.best_validation_accuracy = 0
        # create a string handle to match type of placeholder created before. This train_handle is a iterator over the
        # dataset, and is "wrapped" inside a string_handle()
        train_handle = self.sess.run(self.train_iter.string_handle())
        # initialize iterator for training set. Without this point it is not possible to start reading from dataset
        train_init_op = self.train_iter.initializer
        # feed the handle with the train_handle created before. So:
        # 1. we have created a placeholder handle to prepare the exact input for the net (not numpy arrays but tf.data.Iterators)
        # 2. we created a tf.data.Dataset object and an iterator (self.train_iter) which can be initialized
        # 3. inside train method we prepared an handle to match the placeholder we created before. We initialize the
        #    iterator with self.train_iter.initializer;
        # 4. we fed the handle and the dropout_rate to the model;
        # 5. we start the session with the initialization of the iterator, then we feed the net with the values it needs
        # feed_dict = {self.handle: train_handle, self.dropout_rate: self.conf.dropout_rate, self.is_train: True}

        if self.conf.reload_step > 0:
            self.reload(self.conf.reload_step)
            logging.info('----> Continue Training from step #{}'.format(self.conf.reload_step))
        logging.info('----> Start Training')
        # provide the network the iterators and the dropout_rate we defined before (why the hell? We should put dropout_rate outside)
        self.sess.run(train_init_op)
        for epoch in range(self.global_step_int+1, self.conf.max_epoch):
            feed_dict = {self.handle: train_handle, self.is_train: True}
            # self.is_train = True
            for train_step in range(self.data_reader.num_train_batch):
                summary_time = train_step % self.conf.SUMMARY_FREQ == 0
                last_step = train_step == self.data_reader.num_train_batch - 1
                not_first_step = train_step > 0
                # if train_step > 0 and (train_step % self.conf.SUMMARY_FREQ == 0 or train_step == self.data_reader.num_train_batch-1):
                if not_first_step and (summary_time or last_step):
                    # the second condition is needed to save the summary at the last train_step
                    _, _, _, summary = self.sess.run([self.train_op,
                                                      self.mean_loss_op,
                                                      self.mean_accuracy_op,
                                                      self.merged_summary], feed_dict=feed_dict)
                    loss, acc = self.sess.run([self.mean_loss, self.mean_accuracy])
                    global_step = (epoch - 1) * self.data_reader.num_train_batch + train_step
                    self.save_summary(summary, global_step, mode='train')
                    logging.info('epoch {0}|{1:.01%},\t\ttrain_loss: {2:.4f}, train_acc: {3:.01%}'
                          .format(epoch, (train_step+1)/self.data_reader.num_train_batch, loss, acc))
                else:
                    _, _, _ = self.sess.run([self.train_op, self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)
            if epoch % self.conf.VAL_FREQ == 0:
                self.evaluate(epoch)

    def evaluate(self, epoch):
        self.sess.run(tf.local_variables_initializer())
        val_handle = self.sess.run(self.val_iter.string_handle())
        val_init_op = self.val_iter.initializer
        feed_dict = {self.handle: val_handle, self.is_train: False}
        for step in range(self.data_reader.val_batch_size):
            # as in train method
            self.sess.run(val_init_op)
            self.sess.run([self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)

        valid_loss, valid_acc = self.sess.run([self.mean_loss, self.mean_accuracy])
        summary_valid = self.sess.run(self.merged_summary, feed_dict=feed_dict)
        self.save_summary(summary_valid, epoch * self.data_reader.num_train_batch, mode='valid')
        if valid_acc > self.best_validation_accuracy:
            self.best_validation_accuracy = valid_acc
            self.save(epoch)
            improved_str = '(improved)'
        else:
            improved_str = ''
        logging.info('After {0} epoch: val_loss: {1:.4f}, val_acc: {2:.01%} {3}'.
              format(epoch, valid_loss, valid_acc, improved_str))

    def test(self, epoch_num):
        self.reload(epoch_num)
        self.data_reader = CropsGenerator(self.conf, self.HammingSet)
        self.is_train = False
        self.sess.run(tf.local_variables_initializer())
        for step in range(self.data_reader.num_test_batch):
            x_test, y_test = self.data_reader.generate(mode='test')
            feed_dict = {self.x: x_test, self.y: y_test, self.dropout_rate: 1}
            self.sess.run([self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)
        test_loss, test_acc = self.sess.run([self.mean_loss, self.mean_accuracy])
        logging.info('test_loss= {0:.4f}, test_acc={1:.01%}'.
              format(test_loss, test_acc))

    def save(self, epoch):
        logging.info('----> Saving the model after epoch #{0}'.format(epoch))
        checkpoint_path = os.path.join(self.conf.modeldir+self.conf.run_name, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=epoch)

    def reload(self, epoch):
        model_path, _ = self.check_if_model_exists(epoch)
        logging.info('----> Restoring the model...')
        self.saver.restore(self.sess, model_path)
        logging.info('----> Model-{} successfully restored'.format(epoch))

    def check_if_model_exists(self, epoch: int) -> (str, int):
        """
        Checks if the model indicated in conf.reload_step exists. It raises a ValueError exception
        :param epoch: epoch from which the user want to resume training
        :return: the epoch if it exists. Raises an exception if it does not
        """
        checkpoint_path = os.path.join(self.conf.modeldir + self.conf.run_name)
        if epoch is not 0:
            model_path = os.path.join(checkpoint_path, self.conf.model_name + '-' + str(epoch))
            if not os.path.exists(model_path + '.meta'):
                # print('----> No such checkpoint found', model_path)
                raise ValueError('----> No such checkpoint found: ', model_path)
            return model_path, epoch
        else:
            if not os.path.isdir(checkpoint_path) or not os.listdir(checkpoint_path):
                return None, epoch  # dir does not exist
            else:
                raise FileExistsError('"{}" is not empty. Please delete or move existing files to avoid overwrites.'.format(checkpoint_path))

