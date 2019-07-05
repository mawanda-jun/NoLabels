"""
Copyright 2017-2022 Department of Electrical and Computer Engineering
University of Houston, TX/USA
**********************************************************************************
Author:   Aryan Mobiny
Date:     8/1/2017
Comments: Includes function which creates the 3D ResNet with 50 layer.
        For more information regarding the structure of the network,
        please refer to Table 1 of the original paper:
        "Deep Residual Learning for Image Recognition"
**********************************************************************************
"""

from nets.AlexNet.ops import conv_2d, flatten_layer, fc_layer, dropout, max_pool, lrn, batch_norm_wrapper
from tensorflow.python.keras import layers


def AlexNet(X, rate, is_train):
    """
    It is important to call the function since the "reuse" method is not working in keras yet. This way it is supposed
    to call the class for each layer and to share weights. I stated that since the net is not exploding while calling
    the methods in this way.
    :param X:
    :param rate:
    :param is_train:
    :return:
    """
    net = conv_2d(X, 11, 2, 96, 'CONV1', is_train=is_train, padding="VALID")
    net = max_pool(net, 3, 2, 'MaxPool1')
    net = lrn(net)

    net = conv_2d(net, 5, 2, 256, 'CONV2', is_train=is_train)
    net = max_pool(net, 3, 2, 'MaxPool2')
    net = lrn(net)

    net = conv_2d(net, 3, 1, 384, 'CONV3', is_train=is_train)
    net = conv_2d(net, 3, 1, 384, 'CONV4', is_train=is_train)
    net = conv_2d(net, 3, 1, 256, 'CONV5', is_train=is_train)
    net = max_pool(net, 3, 2, 'MaxPool3')

    layer_flat = flatten_layer(net)
    net = fc_layer(layer_flat, 512, 'FC6', is_train=is_train)

    net = dropout(net, rate, is_train)

    return net

    # net = layers.Conv2D(96, 11, 2, 'same', activation='relu', name='CONV1')(X)
    # # net = layers.BatchNormalization()(net, training=is_train)
    # net = lrn(net)
    # net = layers.MaxPool2D(3, 2, 'same', name='MaxPool1')(net)
    #
    # net = layers.Conv2D(256, 5, 2, 'same', activation='relu', name='CONV2')(net)
    # # net = layers.BatchNormalization()(net, training=is_train)
    # net = lrn(net)
    # net = layers.MaxPool2D(3, 2, 'same', name='MaxPool2')(net)
    #
    # net = layers.Conv2D(384, 3, 1, 'same', activation='relu', name='CONV3')(net)
    # net = layers.Conv2D(384, 3, 1, 'same', activation='relu', name='CONV4')(net)
    # net = layers.Conv2D(256, 3, 1, 'same', activation='relu', name='CONV5')(net)
    # net = layers.MaxPool2D(3, 2, 'same', name='MaxPool3')(net)
    #
    # net = layers.Flatten()(net)
    # net = layers.Dense(512, activation='relu', name='FC6')(net)
    # net = layers.Dropout(rate)(net, training=is_train)
    # return net


def AlexNet_target_task(X, keep_prob, num_cls):
    net = conv_2d(X, 11, 2, 96, 'CONV1', trainable=False)
    net = lrn(net)
    net = max_pool(net, 3, 2, 'MaxPool1')
    net = conv_2d(net, 5, 2, 256, 'CONV2', trainable=False)
    net = lrn(net)
    net = max_pool(net, 3, 2, 'MaxPool2')
    net = conv_2d(net, 3, 1, 384, 'CONV3', trainable=False)
    net = conv_2d(net, 3, 1, 384, 'CONV4', trainable=False)
    net = conv_2d(net, 3, 1, 256, 'CONV5', trainable=False)
    net = max_pool(net, 3, 2, 'MaxPool3')
    layer_flat = flatten_layer(net)
    net = fc_layer(layer_flat, 512, 'FC_1', trainable=True)
    net = dropout(net, keep_prob)
    net = fc_layer(net, num_cls, 'FC_2', trainable=True, use_relu=False)
    return net