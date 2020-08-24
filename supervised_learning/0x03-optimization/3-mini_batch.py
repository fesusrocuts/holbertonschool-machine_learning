#!/usr/bin/env python3
""" 3. Mini-Batch
function def train_mini_batch(X_train, Y_train, X_valid,
Y_valid, batch_size=32, epochs=5, load_path="/tmp/model.ckpt",
save_path="/tmp/model.ckpt"): that trains a loaded neural
network model using mini-batch gradient descent
"""

import numpy as np
import tensorflow as tf
calculate_accuracy = __import__('3-all').calculate_accuracy
calculate_loss = __import__('3-all').calculate_loss
create_placeholders = __import__('3-all').create_placeholders
create_train_op = __import__('3-all').create_train_op
forward_prop = __import__('3-all').forward_prop
shuffle_data = __import__('2-shuffle_data').shuffle_data

def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """ 3. Mini-Batch
    function def train_mini_batch(X_train, Y_train, X_valid,
    Y_valid, batch_size=32, epochs=5, load_path="/tmp/model.ckpt",
    save_path="/tmp/model.ckpt"): that trains a loaded neural
    network model using mini-batch gradient descent
    """
    with tf.Session() as sess:
        s = tf.train.import_meta_graph(load_path+'.meta')
        s.restore(sess, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]
        train_cost = sess.run(loss, feed_dict={x : X_train, y : Y_train})
        train_accuracy = sess.run(accuracy, feed_dict={x : X_train, y : Y_train})
        valid_cost = sess.run(loss, feed_dict={x : X_valid, y : Y_valid})
        valid_accuracy = sess.run(accuracy, feed_dict={x : X_valid, y : Y_valid})
        m = X_train.shape[0]
        X_shuffled = X_train
        Y_shuffled = Y_train
        X_batch = np.zeros((32, X_train.shape[1]))
        Y_batch = np.zeros((32, Y_train.shape[1]))
        for j in range(epochs + 1):
            print("After {} epochs:".format(j))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))
            counter = 0
            for step_number in range(0, int(m/32)):
                for a in range(32):
                    X_batch[a] = X_shuffled[step_number * 32 + a]
                    Y_batch[a] = Y_shuffled[step_number * 32 + a]
                step_accuracy = sess.run(accuracy, feed_dict={x: X_batch, y: Y_batch})
                step_cost = sess.run(loss, feed_dict={x: X_batch, y: Y_batch})
                if (counter == 100):
                    print("\tStep {}:".format(step_number))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy: {}".format(step_accuracy))
                    counter = 0
                counter = counter + 1
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})
            resto = m % 32
            step_number += 1
            X_resto = np.zeros((resto, X_train.shape[1]))
            Y_resto = np.zeros((resto, Y_train.shape[1]))
            for a in range(resto):
                X_resto[a] = X_shuffled[step_number * 32 + a]
                Y_resto[a] = Y_shuffled[step_number * 32 + a]
            step_accuracy = sess.run(accuracy, feed_dict={x: X_resto, y: Y_resto})
            step_cost = sess.run(loss, feed_dict={x: X_resto, y: Y_resto})
            sess.run(train_op, feed_dict={x: X_resto, y: Y_resto})
            X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)
            train_accuracy = step_accuracy
            train_cost = step_cost
            valid_accuracy = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
        save_path = s.save(sess, save_path)
    return(save_path)
