import tensorflow as tf


def cross_entropy_loss(labels_tensor, logits_tensor):
    """
     Calculates the cross-entropy loss function for the given parameters.
     Note that this is for multi-task classification problem.
    :param labels_tensor: Tensor of correct predictions of size [batch_size, numClasses]
    :param logits_tensor: Predicted scores (logits) by the model.
            It should have the same dimensions as labels_tensor
    :return: Cross-entropy Loss tensor
    """
    # use softmax_cross_entropy_with_logits instead of softmax_cross_entropy_with_logits_v2 because our labels are
    # mutually exclusive and one class at a time.
    diff = tf.losses.softmax_cross_entropy(logits=logits_tensor, onehot_labels=labels_tensor)
    # diff = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_tensor, labels=labels_tensor)
    loss = tf.reduce_mean(diff)
    return loss
