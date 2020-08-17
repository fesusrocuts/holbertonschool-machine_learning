#!/usr/bin/env python3
""" 7. Evaluate
function def evaluate(X, Y, save_path):
that evaluates the output of a neural network:
X is a numpy.ndarray containing the input data to evaluate
Y is a numpy.ndarray containing the one-hot labels for X
save_path is the location to load the model from
You are not allowed to use tf.saved_model
Returns: the network’s prediction, accuracy, and loss, respectively
"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """ fn def evaluate(X, Y, save_path):
    that evaluates the output of a neural network:
    X is a numpy.ndarray containing the input data to evaluate
    Y is a numpy.ndarray containing the one-hot labels for X
    save_path is the location to load the model from
    You are not allowed to use tf.saved_model
    Returns: the network’s prediction, accuracy, and loss, respectively
    """
    session = tf.Session()
    saver = tf.train.import_meta_graph(save_path + ".meta")
    checkpoint = tf.train.latest_checkpoint('./')
    saver.restore(session, checkpoint)

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("y:0")
    layer = graph.get_tensor_by_name("layer_2/BiasAdd:0")
    loss = graph.get_tensor_by_name("softmax_cross_entropy_loss/value:0")
    accuracy = graph.get_tensor_by_name("Mean:0")
    layer = session.run(layer, {x: X, y: Y})
    accuracy = session.run(accuracy, {x: X, y: Y})
    loss = session.run(loss, {x: X, y: Y})
    return(layer, accuracy, loss)
