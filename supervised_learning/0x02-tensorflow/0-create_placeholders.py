#!/usr/bin/env python3
""" 0. Placeholders 
function def create_placeholders(nx, classes): that returns
two placeholders, x and y, for the neural network
nx: the number of feature columns in our data
classes: the number of classes in our classifier
Returns: placeholders named x and y, respectively
x is the placeholder for the input data to the neural network
y is the placeholder for the one-hot labels for the input data
"""
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

def create_placeholders(nx, classes):
    """ fn create_placeholders,
    nx: the number of feature columns in our data
    classes: the number of classes in our classifier
    Returns: placeholders named x and y, respectively
    x is the placeholder for the input data to the neural network
    y is the placeholder for the one-hot labels for the input data
    """
    x = tf.placeholder(tf.float64, [None, nx], name='x')
    y = tf.placeholder(tf.float64, [None, classes], name='y')
    return(x, y)
