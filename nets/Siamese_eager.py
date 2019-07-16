import tensorflow as tf
from tensorflow.python.keras import layers, regularizers
from inspect import signature
l2 = regularizers.l2()
tf.enable_eager_execution()


# class AlexNet(tf.keras.layers.Layer):
#     """
#     Create a super-layer made up of "Alex". self.alexnet is a list of layers to be built
#     """
#     def __init__(self, name, **kwargs):
#         super(AlexNet, self).__init__(name=name, **kwargs)
#         self.alexnet = [
#             layers.Conv2D(96, 11, 2, 'same', activation='relu', name='CONV1'),
#             layers.BatchNormalization(momentum=0.9, name="batch_norm_1"),
#             layers.MaxPool2D(3, 2, 'valid', name='MaxPool1'),
#
#             layers.Conv2D(256, 5, 2, 'same', activation='relu', name='CONV2'),
#             layers.BatchNormalization(momentum=0.9, name="batch_norm_2"),
#             layers.MaxPool2D(3, 2, 'valid', name='MaxPool2'),
#
#             layers.Conv2D(384, 3, 1, 'same', activation='relu', name='CONV3'),
#             layers.Conv2D(384, 3, 1, 'same', activation='relu', name='CONV4'),
#             layers.Conv2D(256, 3, 1, 'same', activation='relu', name='CONV5'),
#             layers.BatchNormalization(momentum=0.9, name="batch_norm_3"),
#             layers.MaxPool2D(3, 2, 'valid', name='MaxPool3'),
#
#             layers.Flatten(),
#             layers.Dense(512, activation='relu', name='FC6'),
#             # layers.BatchNormalization(momentum=0.99, name="batch_norm_2"),
#             layers.Dropout(0.5)
#         ]
#
#     def call(self, inputs, training=None, mask=None):
#         """
#         Define call function: this is what is called when the object is called in that way:
#         i = <instance_of_object>
#         result = i(x)
#         :param inputs:
#         :param training:
#         :param mask:
#         :return:
#         """
#         x = self.alexnet[0](inputs)  # compute first output
#         for layer in self.alexnet[1:]:
#             # invoke every layer with the output. We need to check the signature of every layer: dropout and batch_norm
#             # accept also the 'training' parameter, which deactivate the layer in case of validation or test
#             if 'training' not in str(signature(layer.call)):
#                 x = layer(x)
#             else:
#                 x = layer(x, training=training)
#         return x


class Siamese(tf.keras.Model):
    """
    Define a Siamese object, on which we can do inference and prediction.
    """
    def __init__(self, conf, **kwargs):
        super(Siamese, self).__init__(**kwargs)
        self.num_crops = conf.numCrops
        self.tile_size = conf.tileSize
        self.num_channels = conf.numChannels
        self.num_classes = conf.hammingSetSize

        self.conv1 = layers.Conv2D(96, 11, 2, 'same', activation='relu', name='CONV1')
        self.bn1 = layers.BatchNormalization(momentum=0.99, name="batch_norm_1")
        self.maxpool1 = layers.MaxPool2D(3, 2, 'valid', name='MaxPool1')

        self.conv2 = layers.Conv2D(256, 5, 2, 'same', activation='relu', name='CONV2')
        self.bn2 = layers.BatchNormalization(momentum=0.99, name="batch_norm_2")
        self.maxpool2 = layers.MaxPool2D(3, 2, 'valid', name='MaxPool2')

        self.conv3 = layers.Conv2D(384, 3, 1, 'same', activation='relu', name='CONV3')
        self.conv4 = layers.Conv2D(384, 3, 1, 'same', activation='relu', name='CONV4')
        self.conv5 = layers.Conv2D(256, 3, 1, 'same', activation='relu', name='CONV5')
        self.bn3 = layers.BatchNormalization(momentum=0.99, name="batch_norm_3")
        self.maxpool3 = layers.MaxPool2D(3, 2, 'valid', name='MaxPool3')
        self.flatten = layers.Flatten()
        self.fc6 = layers.Dense(512, activation='relu', name='FC6', kernel_regularizer=l2, bias_regularizer=l2)
        self.alex_dropout = layers.Dropout(0.5, name='dropout_alexnet')

        # define out-of-block layers
        self.dense = layers.Dense(4096, activation='relu', name='FC7')
        self.bn = layers.BatchNormalization(momentum=0.995, name='batch_norm_last')
        self.dropout = layers.Dropout(0.5, name='last_dropout')
        self.classifier = layers.Dense(self.num_classes, activation='softmax', name='FC8')

    def call(self, inputs, training=None, mask=None):
        """
        inputs is now a stack of tiles. we need to unstack them, to feed the self.alex and to collect all outputs
        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        # define Alex Block as a list of <num_crops> Alexes
        alex_block = [
            self.conv1, self.bn1, self.maxpool1,
            self.conv2, self.bn2, self.maxpool2,
            self.conv3, self.conv4, self.conv5, self.bn3, self.maxpool3,
            self.flatten, self.fc6, self.alex_dropout
        ]
        x = tf.unstack(inputs, axis=-1)
        alexes = []
        for i in range(self.num_crops):
            # it should reuse the variables inside alex_block
            alexes.append(self.call_alex(alex_block, x[i], training))
        siamese_block = tf.concat(alexes, axis=1)

        x = self.dense(siamese_block)
        x = self.bn(x, training=training)
        x = self.dropout(x, training=training)
        out = self.classifier(x)
        return out

    @staticmethod
    def call_alex(alex_block, inputs, training):
        x = alex_block[0](inputs)  # compute first output
        for layer in alex_block[1:]:
            # invoke every layer with the output. We need to check the signature of every layer: dropout and batch_norm
            # accept also the 'training' parameter, which deactivate the layer in case of validation or test
            if 'training' not in str(signature(layer.call)):
                x = layer(x)
            else:
                x = layer(x, training=training)
        return x

    def summary(self, line_length=None, positions=None, print_fn=None):
        x = tf.keras.Input(shape=(self.tile_size, self.tile_size, self.num_channels, self.num_crops))
        tf.keras.Model(inputs=x, outputs=self.call(x, training=True)).summary(line_length, positions, print_fn)


if __name__ == '__main__':
    from config import conf
    inputs = tf.keras.Input(shape=(conf.tileSize, conf.tileSize, conf.numChannels, conf.numCrops))
    model = Siamese(conf)
    model.build(input_shape=inputs.shape)
    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    model.summary(line_length=110)

