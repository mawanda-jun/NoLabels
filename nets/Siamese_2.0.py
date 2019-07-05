import tensorflow as tf
tf.enable_eager_execution()
from tensorflow.python.keras import layers, losses, models
from typing import List, Dict, Tuple
import numpy as np


# placeholder for params
class Params:
    @property
    def dropout_rate(self):
        return None


params = Params()
shared_layers = {
    'CONV1': layers.Conv2D(96, 11, 2, 'same', activation='relu', name='CONV1'),
    'FC6': layers.Dense(512, activation='relu', name="FC6")
}

X_train = np.random.random((3, 256, 256, 3, 9))
y_train = np.random.random((3, 5))
X_test = np.random.random((3, 256, 256, 3, 9))
y_test = np.random.random((3, 5))

batch_size = 2
epochs = 2
num_classes = 10


class AlexNet(tf.keras.Model):
    """
    Create a super-layer made up of "Alex". Alex is now a callable layer
    """
    def __init__(self, dropout_rate):
        super(AlexNet, self).__init__()
        self.alexnet = models.Sequential([
            layers.Conv2D(96, 11, 2, 'same', activation='relu', name='CONV1'),
            layers.MaxPool2D(3, 2, 'same', name='MaxPool1'),

            layers.Conv2D(256, 5, 2, 'same', activation='relu', name='CONV2'),
            layers.MaxPool2D(3, 2, 'same', name='MaxPool2'),

            layers.Conv2D(384, 3, 1, 'same', activation='relu', name='CONV3'),
            layers.Conv2D(384, 3, 1, 'same', activation='relu', name='CONV4'),
            layers.Conv2D(256, 3, 1, 'same', activation='relu', name='CONV5'),
            layers.MaxPool2D(3, 2, 'same', name='MaxPool3'),

            layers.Flatten(),
            layers.Dense(512, activation='relu', name='FC6'),
            layers.Dropout(dropout_rate)
        ])

    def call(self, inputs, training=None, mask=None):
        return self.alexnet(inputs, training)


class Siamese(tf.keras.Model):
    def __init__(self, dropout_rate, num_classes):
        super(Siamese, self).__init__()
        self.alex = AlexNet(dropout_rate)
        self.alex_block = [self.alex for _ in range(9)]
        self.siamese = models.Sequential()
        self.classifier = layers.Dense(num_classes, name='FC8')
        self.dropout_rate = dropout_rate

    def call(self, inputs, training=None, mask=None):
        """
        inputs is now a stack of tiles. we need to unstack them, to feed the self.alex and to collect all outputs
        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        x = tf.unstack(inputs, axis=-1)
        siamese = []
        for i in range(9):
            # it should reuse the variables inside alex_block
            siamese.append(self.alex_block[i](x[i], training))
        # siamese_block = layers.Concatenate(siamese)
        siamese_block = tf.concat(siamese, axis=1)

        self.siamese.add([
            layers.Dense(4096, activation='relu',  name='FC7'),
            layers.Dropout(self.dropout_rate),
            self.classifier
        ])
        return self.siamese(siamese_block, training=training)


if __name__ == '__main__':
    model = Siamese(dropout_rate=0.5, num_classes=num_classes)
    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    dummy_x = tf.zeros((1, 256, 256, 3, 9))
    model._set_inputs(dummy_x)
    model.summary()

