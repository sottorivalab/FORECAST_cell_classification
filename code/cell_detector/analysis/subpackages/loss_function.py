import tensorflow as tf


def loss_function(network):
    logits = network.logits
    labels = network.labels
    epsilon = 1e-6
    with tf.name_scope('Cost_Function'):
        # CostFunction
        clipped_logits = tf.clip_by_value(logits, epsilon, 1.0 - epsilon)
        cross_entropy = tf.reduce_sum(
            -labels * tf.log(clipped_logits) - (1.0 - labels) * tf.log(1.0 - clipped_logits))
        network.loss = cross_entropy
        _ = tf.summary.scalar('Loss', network.loss)

        return network.loss