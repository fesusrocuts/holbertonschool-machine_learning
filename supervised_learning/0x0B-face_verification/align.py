#!/usr/bin/env python3
"""
FaceAlign Class
"""
import numpy as np
import cv2
import dlib


class FaceAlign:
    """
    FaceAlign Class
    """
    def __init__(self, shape_predictor_path):
        """
        class constructor def __init__(self, shape_predictor_path):
        shape_predictor_path is the path to the dlib shape
        predictor model
        Sets the public instance attributes:
        detector - contains dlib‘s default face detector
        shape_predictor - contains the dlib.shape_predictor
        """
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)

    def detect(self, image):
        """
        public instance method def detect(self, image):
        that detects a face in an image:
        image is a numpy.ndarray of rank 3 containing an image
        from which to detect a face
        Returns: a dlib.rectangle containing the boundary
        box for the face in the image, or None on failure
        If multiple faces are detected, return the dlib.rectangle
        with the largest area
        If no faces are detected, return a dlib.rectangle that is
        the same as the image
        """
        try:
            faces = self.detector(image, 1)
            area = 0

            for iter in faces:
                if iter.area() > area:
                    area = iter.area()
                    rectangle = iter

            if area == 0:
                rectangle = (dlib.rectangle(left=0, top=0,
                                            right=image.shape[1],
                                            botom=image.shape[0]))
            return rectangle
        except Exception as e:
            return None

    def find_landmarks(self, image, detection):
        """
        public instance method 
        def find_landmarks(self, image, detection):
        that finds facial landmarks:
        image is a numpy.ndarray of an image from which
        to find facial landmarks
        detection is a dlib.rectangle containing the
        boundary box of the face in the image
        Returns: a numpy.ndarray of shape (p, 2)containing
        the landmark points, or None on failure
        p is the number of landmark points
        2 is the x and y coordinates of the point
        """
        landmark_points = self.shape_predictor(image, detection)
        if not landmark_points:
            return None

        coor_points = np.zeros((68, 2), dtype='int')
        for m in range(0, 68):
            coor_points[m] = [landmark_points.part(m).x, landmark_points.part(m).y]
        return coor_points

    def align(self, image, landmark_indices, anchor_points, size=96):
        """
        public instance method
        def align(self, image, landmark_indices,
        anchor_points, size=96): that aligns an image for
        face verification:
        image is a numpy.ndarray of rank 3 containing the
        image to be aligned
        landmark_indices is a numpy.ndarray of shape (3,)
        containing the indices of the three landmark points
        that should be used for the affine transformation
        anchor_points is a numpy.ndarray of shape (3, 2)
        containing the destination points for the affine
        transformation, scaled to the range [0, 1]
        size is the desired size of the aligned image
        Returns: a numpy.ndarray of shape (size, size, 3)
        containing the aligned image, or None if no face is detected
        """
        rectangle_box = self.detect(image)
        coor_points = self.find_landmarks(image, rectangle_box)
        coor_points = coor_points.astype('float32')
        face_anchors = anchor_points * size
        M = cv2.getAffineTransform(coordPts, face_anchors)
        img = cv2.warpAffine(image, M, (size, size))

        return img
