import tensorflow as tf
from tensorflow.python.keras import layers
from inspect import signature

tf.enable_eager_execution()


class SiameseFT(tf.keras.Model):
    """
    Define a Siamese object, on which we can do inference and prediction.
    """
    def __init__(self, conf, **kwargs):
        super(SiameseFT, self).__init__(**kwargs)

        self.conf = conf

        self.conv1 = layers.Conv2D(96, 11, 2, 'same', activation='relu', name='CONV1')
        self.batch_norm_1 = layers.BatchNormalization(momentum=0.9, name="batch_norm_1")
        self.max_pool_1 = layers.MaxPool2D(3, 2, 'same', name='MaxPool1')

        self.conv2 = layers.Conv2D(256, 5, 2, 'same', activation='relu', name='CONV2')
        self.batch_norm_2 = layers.BatchNormalization(momentum=0.9, name="batch_norm_2")
        self.max_pool_2 = layers.MaxPool2D(3, 2, 'same', name='MaxPool2')

        self.conv3 = layers.Conv2D(384, 3, 1, 'same', activation='relu', name='CONV3')
        self.conv4 = layers.Conv2D(384, 3, 1, 'same', activation='relu', name='CONV4')
        self.conv5 = layers.Conv2D(256, 3, 1, 'same', activation='relu', name='CONV5')
        self.max_pool_3 = layers.MaxPool2D(3, 2, 'same', name='MaxPool3')

        self.flatten = layers.Flatten()
        self.dense_1 = layers.Dense(512, activation='relu', name='FC1')
        self.dropout = layers.Dropout(0.5)
        self.dense_2 = layers.Dense(conf.num_clas, activation='softmax', name='FC2')

    def call(self, inputs, training=None, mask=None):
        """
        inputs is now a stack of tiles. we need to unstack them, to feed the self.alex and to collect all outputs
        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        x = self.conv1(inputs)
        x = self.batch_norm_1(x, training)
        x = self.max_pool_1(x)

        x = self.conv2(x)
        x = self.batch_norm_2(x, training)
        x = self.max_pool_2(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.max_pool_3(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dropout(x, training)
        return self.dense_2(x)


if __name__ == '__main__':
    model = SiameseFT(num_clas=101)
    dummy_x = tf.zeros((1, 256, 256, 3))
    # weights = "D:\Repo\NoLabels\ResultsJPS\run01\model_dir\weights.01-7.31.hdf5"
    model._set_inputs(dummy_x)

    # freezed of the first 13 layers
    freezed_layers = ['CONV1', 'CONV2', 'CONV3', 'CONV4', 'CONV5']
    for layer in model.layers[:13]:
        if freezed_layers.__contains__(layer.name):
            layer.trainable = False

    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    model.load_weights(weights, 9)
    model.summary()
