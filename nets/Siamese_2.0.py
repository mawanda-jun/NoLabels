import tensorflow as tf
tf.enable_eager_execution()
from tensorflow.python.keras import layers
from inspect import signature


class AlexNet(tf.keras.Model):
    """
    Create a super-layer made up of "Alex". Alex is now a callable layer
    """
    def __init__(self, dropout_rate):
        super(AlexNet, self).__init__()
        self.alexnet = [
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
        ]

    def call(self, inputs, training=None, mask=None):
        x = self.alexnet[0](inputs)
        for layer in self.alexnet[1:]:
            if 'training' not in str(signature(layer.call)):
                x = layer(x)
            else:
                x = layer(x, training)
        return x


class Siamese(tf.keras.Model):
    def __init__(self, dropout_rate, num_classes):
        super(Siamese, self).__init__()
        self.alex = AlexNet(dropout_rate)
        self.alex_block = [self.alex for _ in range(9)]
        self.dense = layers.Dense(4096, activation='relu', name='FC7')
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
        alexes = []
        for i in range(9):
            # it should reuse the variables inside alex_block
            alexes.append(self.alex_block[i](x[i], training))
        siamese_block = tf.concat(alexes, axis=1)

        x = self.dense(siamese_block)
        x = layers.Dropout(self.dropout_rate)(x, training)
        return self.classifier(x)


if __name__ == '__main__':
    model = Siamese(dropout_rate=0.5, num_classes=5)
    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    dummy_x = tf.zeros((1, 256, 256, 3, 9))
    model._set_inputs(dummy_x)
    model.summary()

