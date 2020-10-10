#!/usr/bin/env python3
"""
load_images, load_csv, generate_triplets functions
"""

import csv
import cv2
import numpy as np
import os
import glob


def load_images(images_path, as_array=True):
    """
    Write the function def load_images(images_path, as_array=True):
    that loads images from a directory or file:
    images_path is the path to a directory from which to load images
    as_array is a boolean indicating whether the images should
    be loaded as one numpy.ndarray
        If True, the images should be loaded as a numpy.ndarray
        of shape (m, h, w, c) where:
            m is the number of images
            h, w, and c are the height, width, and number of
            channels of all images, respectively
        If False, the images should be loaded as a list of
        individual numpy.ndarrays
    All images should be loaded in RGB format
    The images should be loaded in alphabetical order by filename
    Returns: images, filenames
        images is either a list/numpy.ndarray of all images
        filenames is a list of the filenames associated with
        each image in images
    """
    # image path
    image_paths = glob.glob(images_path + "/*")

    # get image names
    image_names = []
    for imPath in image_paths:
        image_names.append(imPaths.split('/')[-1])

    # sort
    imageIndex = np.argsort(image_names)

    # read and color images, pics or photos 
    image_original = []
    for photos in image_paths:
        image_original.append(cv2.imread(photos))
    for photos in image_original:
        image_original.append(cv2.cvtColor(photos, cv2.COLOR_BGR2RGB))

    images = []
    file_names = []

    # append images and file names
    for m in imageIndex:
        images.append(imageOrig[m])
        file_names.append(image_names[m])

    # as_array
    if as_array:
        images = np.stack(images, axis=0)

    return images, file_names


def load_csv(csv_path, params={}):
    """
    Also in utils.py, write a function
    def load_csv(csv_path, params={}):
    that loads the contents of a csv file as a list of lists:
    csv_path is the path to the csv to load
    params are the parameters to load the csv with
    Returns: a list of lists representing the contents found
    in csv_path
    """
    csv_list = []
    with open(csv_path, 'r') as csv_file:
        csvReader = csv.reader(csv_file, params)
        for item in csvReader:
            csv_list.append(item)
    return csv_list


def save_images(path, images, filenames):
    """
    write a function
    def save_images(path, images, filenames):
    that saves images to a specific path:
    path is the path to the directory in which the
    images should be saved
    images is a list/numpy.ndarray of images to save
    filenames is a list of filenames of the images to save
    Returns: True on success and False on failure
    """
    if os.path.exists(path):
        for img, image_name in zip(images, filenames):
            photo = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite('./' + path + '/' + image_name, photo)
        return True
    else:
        return False


def generate_triplets(images, filenames, triplet_names):
    """
    write a function
    def generate_triplets(images, filenames, triplet_names):
    that generates triplets:
    images is a numpy.ndarray of shape (i, n, n, 3) containing
    the aligned images in the dataset
        i is the number of images
        n is the size of the aligned images
    filenames is a list of length i containing the corresponding
    filenames for images
    triplet_names is a list of length m of lists where each
    sublist contains the filenames of an anchor, positive,
    and negative image, respectively
        m is the number of triplets
    Returns: a list [A, P, N]
        A is a numpy.ndarray of shape (m, n, n, 3)
            containing the anchor images for all m triplets
        P is a numpy.ndarray of shape (m, n, n, 3)
            containing the positive images for all m triplets
        N is a numpy.ndarray of shape (m, n, n, 3)
            containing the negative images for all m triplets
    """
    image_names = [filenames[m].split('.')[0] for m in range(len(filenames))]

    anchor_names = [names[0] for names in triplet_names]
    positive_names = [names[1]for names in triplet_names]
    negative_names = [names[2]for names in triplet_names]

    a_img = [m for m in range(len(image_names)) if image_names[m] in anchor_names]
    p_img = [m for m in range(len(image_names)) if image_names[m] in positive_names]
    n_img = [m for m in range(len(image_names)) if image_names[m] in negative_names]

    A = images[a_img]
    P = images[p_img]
    N = images[n_img]

    # print(A.shape)
    # print(P.shape)
    # print(N.shape)

    return [A, P, N]
