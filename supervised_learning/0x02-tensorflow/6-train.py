#!/usr/bin/env python3
""" 6. Train
function def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
activations, alpha, iterations, save_path="/tmp/model.ckpt"):
that builds, trains, and saves a neural network classifier:
X_train is a numpy.ndarray containing the training input data
Y_train is a numpy.ndarray containing the training labels
X_valid is a numpy.ndarray containing the validation input data
Y_valid is a numpy.ndarray containing the validation labels
layer_sizes is a list containing the number of nodes
in each layer of the network
actications is a list containing the activation functions
for each layer of the network
alpha is the learning rate
iterations is the number of iterations to train over
save_path designates where to save the model
"""
import tensorflow as tf
create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """ fn def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
    activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    that builds, trains, and saves a neural network classifier:
    X_train is a numpy.ndarray containing the training input data
    Y_train is a numpy.ndarray containing the training labels
    X_valid is a numpy.ndarray containing the validation input data
    Y_valid is a numpy.ndarray containing the validation labels
    layer_sizes is a list containing the number of nodes
    in each layer of the network
    actications is a list containing the activation functions
    for each layer of the network
    alpha is the learning rate
    iterations is the number of iterations to train over
    save_path designates where to save the model
    """
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    loss = create_train_op(loss, alpha)
    loss_v = create_train_op(loss, alpha)

    saver = tf.train.Saver()
    initialize = tf.global_variables_initializer()
    session = tf.Session()
    session.run(initialize)
    for iter in range(iterations + 1):
        cost = session.run(loss, feed_dict={x: X_train, y: Y_train})
        acc = session.run(accuracy, feed_dict={x: X_train, y: Y_train})
        cost_v = session.run(loss, feed_dict={x: X_valid, y: Y_valid})
        acc_v = session.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
        if iter % 100 == 0 or iter == iterations:
            print("After {} iterations:".format(iter))
            print("Training Cost: {}".format(cost))
            print("Training Accuracy: {}".format(acc))
            print("Validation Cost: {}".format(cost_v))
            print("Validation Accuracy: {}".format(acc_v))
        session.run(loss,  feed_dict={x: X_train, y: Y_train})
    saver.save(session, save_path)
    return save_path
