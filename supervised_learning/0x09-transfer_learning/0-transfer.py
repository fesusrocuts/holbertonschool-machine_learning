#!/usr/bin/env python3
"""
Write a python script that trains a convolutional neural network
    to classify the CIFAR 10 dataset:
    You must use one of the applications listed in Keras Applications
    Your script must save your trained model in the current working
        directory as cifar10.h5
    Your saved model should be compiled
    Your saved model should have a validation accuracy of 87% or higher
    Your script should not run when the file is imported
    Hint1: The training and tweaking of hyperparameters may take
        a while so start early!
    Hint2: The CIFAR 10 dataset contains 32x32 pixel images,
        however most of the Keras applications are trained on much
        larger images. Your first layer should be a lambda layer
        that scales up the data to the correct size
    Hint3: You will want to freeze most of the application layers.
        Since these layers will always produce the same output,
        you should compute the output of the frozen layers ONCE
        and use those values as input to train the remaining trainable
        layers. This will save you A LOT of time.

In the same file, write a function def preprocess_data(X, Y):
    that pre-processes the data for your model:

    X is a numpy.ndarray of shape (m, 32, 32, 3) containing
        the CIFAR 10 data, where m is the number of data points
    Y is a numpy.ndarray of shape (m,) containing the CIFAR 10
        labels for X
    Returns: X_p, Y_p
        X_p is a numpy.ndarray containing the preprocessed X
        Y_p is a numpy.ndarray containing the preprocessed Y

NOTE: About half of the points for this project are for the blog post
    in the next task. While you are attempting to train your model,
    keep track of what you try and why so that you have a log to
    reference when it is time to write your report.
"""
import tensorflow.keras as K

# step 1
# Classify ImageNet classes with ResNet50
# from tensorflow.keras.applications.resnet50 import ResNet50
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.resnet50 import
# preprocess_input, decode_predictions

"""
model = K.applications.resnet50.ResNet50(weights='imagenet')

img_path = 'elephant2.jpg'
img = K.preprocessing.image.load_img(img_path, target_size=(224, 224))
x = K.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = K.applications.resnet50.preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', K.applications.resnet50.decode_predictions
    (preds, top=3)[0])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225),
# (u'n01871265', u'tusker', 0.1122357),
# (u'n02504458', u'African_elephant', 0.061040461)]
# Predicted: [('n02504013', 'Indian_elephant', 0.94287986),
# ('n01871265', 'tusker', 0.038615398),
# ('n02504458', 'African_elephant', 0.018336201)]
"""


def load_data():
    """
    load data: cifar10
    """
    # import data
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()

    # Normalize values to range between 0 and 1
    # Change integers to floats
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    # X_train = X_train / 255
    # X_test = X_test / 255

    # one hot target values
    Y_train = K.utils.to_categorical(Y_train, 10)
    Y_test = K.utils.to_categorical(Y_test, 10)
    return X_train, Y_train, X_test, Y_test


def model_definition(Y_train):
    """
    model based on resnet50 architecture for CIFAR10
    """
    # use resnet50 for bottlensck features
    model = K.applications.resnet50.ResNet50(include_top=False,
                                             weights='imagenet',
                                             input_tensor=K.Input(
                                                 shape=(32, 32, 3)),
                                             classes=Y_train.shape[1])
    # X_p = K.applications.resnet50.preprocess_input(X, data_format=None)
    for layer in model.layers:
        # get layers before fully connected layers
        if (layer.name[0:5] != 'block'):
            layer.trainiable = False

    modelY = K.Sequential()

    modelY.add(model)
    modelY.add(K.layers.Flatten())
    modelY.add(K.layers.Dense(32, activation='relu',
                              kernel_initializer='he_uniform'))
    modelY.add(K.layers.Dense(10, activation="softmax"))
    modelY.summary()
    # try:
    #     modelY.save_weights('cifar10.h5')
    # except Exception as e:
    #     modelY.save('cifar10.h5')
    # modelY.save_weights('cifar10.h5')
    return modelY


def compile_model(new_cnn):
    """
    compile model separate, model type ResNet50 classes=Y_train.shape[1]
    """
    opt = K.optimizers.SGD(lr=0.001, momentum=0.9)
    new_cnn.compile(optimizer=opt,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return new_cnn


def train_model(new_cnn, X_train, Y_train, X_test, Y_test, batch, epochs):
    """
    train model
    """
    dataGen = K.preprocessing.image.ImageDataGenerator(rotation_range=15,
                                                       width_shift_range=0.1,
                                                       height_shift_range=0.1,
                                                       horizontal_flip=True)
    dataGen.fit(X_train)
    return new_cnn.fit_generator(dataGen.flow(X_train, Y_train,
                                              batch_size=batch),
                                 steps_per_epoch=X_train.shape[0] / batch,
                                 epochs=epochs,
                                 verbose=1,
                                 validation_data=(X_test, Y_test))


def preprocess_data(X, Y):
    """
    X: numpy.ndarray, shape(m, 32, 32, 3) containing CIFAR 10 data
    m: number of data points
    Y: numpy.ndarray, shape(m, ) containing the CIFAR 10 labels for X
    X_p: numpy.ndarray containing preprocessed X
    Y_p: numpy.ndarray containing preprocessed Y
    Return: X_p, Y_p

    trained model:  save in directory as cifar10.h5
                    should be compiled
                    validation accuracy of 88% or higher
    file script should not run when file is imported
    """
    model = K.applications.resnet50.ResNet50(weights='imagenet',
                                             input_tensor=K.Input(
                                                 shape=(32, 32, 3)))
    X_p = K.applications.resnet50.preprocess_input(X, data_format=None)
    X_p_preds = model.predict(X_p)
    # print('Predicted:', K.applications.resnet50.decode_predictions(
    #       X_p_preds, top=3)[0])

    Y_p = K.utils.to_categorical(Y, num_classes=10)
    return (X_p, Y_p)


if __name__ == '__main__':
    # step by step make it
    batch = 50
    epochs = 50

    X_train, Y_train, X_test, Y_test = load_data()
    t_model = model_definition(Y_train)
    t_model = compile_model(t_model)
    # history = train_model(t_model, X_train, Y_train,
    #                        X_test, Y_test, batch, epochs)
    try:
        t_model.save_weights('cifar10.h5')
    except Exception as e:
        t_model.save('cifar10.h5')
