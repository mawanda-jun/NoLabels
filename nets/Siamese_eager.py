import tensorflow as tf
from tensorflow.python.keras import layers
from inspect import signature
tf.enable_eager_execution()


class AlexNet(tf.keras.layers.Layer):
    """
    Create a super-layer made up of "Alex". self.alexnet is a list of layers to be built
    """
    def __init__(self, name, input_shape, **kwargs):
        super(AlexNet, self).__init__(name=name, **kwargs)
        self.alexnet = [
            layers.Conv2D(96, 11, 2, 'same', activation='relu', name='CONV1', input_shape=input_shape),
            layers.BatchNormalization(momentum=0.9, name="batch_norm_1"),
            layers.MaxPool2D(3, 2, 'same', name='MaxPool1'),

            layers.Conv2D(256, 5, 2, 'same', activation='relu', name='CONV2'),
            layers.BatchNormalization(momentum=0.9, name="batch_norm_2"),
            layers.MaxPool2D(3, 2, 'same', name='MaxPool2'),

            layers.Conv2D(384, 3, 1, 'same', activation='relu', name='CONV3'),
            layers.Conv2D(384, 3, 1, 'same', activation='relu', name='CONV4'),
            layers.Conv2D(256, 3, 1, 'same', activation='relu', name='CONV5'),
            layers.MaxPool2D(3, 2, 'same', name='MaxPool3'),

            layers.Flatten(),
            layers.Dense(1024, activation='relu', name='FC6'),
            layers.Dropout(0.5)
        ]

    def call(self, inputs, training=None, mask=None):
        """
        Define call function: this is what is called when the object is called in that way:
        i = <instance_of_object>
        result = i(x)
        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        x = self.alexnet[0](inputs)  # compute first output
        for layer in self.alexnet[1:]:
            # invoke every layer with the output. We need to check the signature of every layer: dropout and batch_norm
            # accept also the 'training' parameter, which deactivate the layer in case of validation or test
            if 'training' not in str(signature(layer.call)):
                x = layer(x)
            else:
                x = layer(x, training)
        return x


class Siamese(tf.keras.Model):
    """
    Define a Siamese object, on which we can do inference and prediction.
    """
    def __init__(self, num_classes, **kwargs):
        super(Siamese, self).__init__(**kwargs)
        self.alex = AlexNet('alex', (-1, 64, 64, 3))
        self.dense = layers.Dense(4096, activation='relu', name='FC7')
        self.bn = layers.BatchNormalization(momentum=0.9, name='Batch_norm_last')
        self.dropout = layers.Dropout(0.5, name='last_dropout')
        self.classifier = layers.Dense(num_classes, activation='softmax', name='FC8')

    def call(self, inputs, training=None, mask=None):
        """
        inputs is now a stack of tiles. we need to unstack them, to feed the self.alex and to collect all outputs
        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        alex_block = [self.alex for _ in range(9)]
        x = tf.unstack(inputs, axis=-1)
        alexes = []
        for i in range(9):
            # it should reuse the variables inside alex_block
            alexes.append(alex_block[i](x[i], training))
        siamese_block = tf.concat(alexes, axis=1)

        x = self.dense(siamese_block)
        x = self.bn(x, training)
        x = self.dropout(x, training)
        out = self.classifier(x)
        return out


if __name__ == '__main__':
    model = Siamese(num_classes=5)
    dummy_x = tf.zeros((1, 64, 64, 3, 9))
    model._set_inputs(dummy_x)
    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    model.summary()

