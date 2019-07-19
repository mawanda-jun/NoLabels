import tensorflow as tf
from tensorflow.python.keras import layers
from inspect import signature

tf.enable_eager_execution()


def loss_op(y_pred, y_true):
    diff = tf.losses.softmax_cross_entropy(y_true, y_pred, from_logits=True)
    return tf.reduce_mean(diff)


class AlexNet(tf.keras.layers.Layer):
    """
    Create a super-layer made up of "Alex". self.alexnet is a list of layers to be built
    """
    def __init__(self, name, **kwargs):
        super(AlexNet, self).__init__(name, **kwargs)
        self.alexnet = []
        self.flatten_out = None
        self.config = {}
        # trainable parameters in form of output shape for every layer in AlexNet
        self.shapes = [
            (11, 11, 3, 96),
            (96,),
            (),
            (5, 5, 3, 256),
            (256, ),
            (),
            (3, 3, 3, 384),
            (3, 3, 3, 384),
            (3, 3, 3, 256),
            (),
            (),
            (256*6*6, 512),
            (),
            (),
        ]

    def build(self, input_shape=(-1, 64, 64, 3)):
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
            layers.Dense(512, activation='relu', name='FC1'),
            layers.Dropout(0.5),
            layers.Dense(self.config.num_clas, activation='relu', name='FC2')
        ]
        for i, layer in enumerate(self.alexnet):
            # adds the weights for model.summary(). These are not going to be trained
            self.add_weight(name=layer.name, shape=self.shapes[i])

    def get_config(self):
        return self.config

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


class SiameseFT(tf.keras.Model):
    """
    Define a Siamese object, on which we can do inference and prediction.
    """
    def __init__(self, conf, **kwargs):
        super(SiameseFT, self).__init__(**kwargs)

        self.conf = conf

        self.conv1 = layers.Conv2D(96, 11, 2, 'same', activation='relu', name='CONV1', input_shape=(-1, 64, 64, 3))
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
