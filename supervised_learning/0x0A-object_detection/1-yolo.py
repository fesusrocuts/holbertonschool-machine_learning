#!/usr/bin/env python3
"""
Yolo Object Detection
"""
import tensorflow.keras as K
import numpy as np


class Yolo():
    """class Yolo"""

    def validateClass(self, listC):
        """function used to remove spaces"""
        arr = []
        for item in listC:
            arr.append(item.strip(" "))
        return arr
    
    def createBoxes(self, grid_height, grid_width, net_box):
        """
        boxes: a list of numpy.ndarrays of
                shape (grid_height, grid_width, anchor_boxes, 4) containing
                the processed boundary boxes for each output, respectively:
                4 => (x1, y1, x2, y2)
                (x1, y1, x2, y2) should represent the boundary box relative
                to original image
        """
        for r in range(grid_height):
                for c in range(grid_width):
                    for b in range(net_box):
                        y, x, w, h, = net_bx[r, c, b, :4]
                        # center image using height and width
                        x = (c + x)
                        ctr_x = x / grid_width
                        y = (r + y)
                        ctr_y = y / grid_height
                        # image height and width
                        w = (anchors[b][0] * np.exp(w))
                        im_width = w / input_w
                        h = (anchors[b][1] * np.exp(h))
                        im_height = h / input_h

                        # define scale
                        x_Box = (ctr_x - im_width/2) * image_width
                        y_Box = (ctr_y - im_height/2) * image_height
                        x_2Box = (ctr_x + im_width/2) * image_width
                        y_2Box = (ctr_y + im_width/2) * image_height

                        # can use BoundBox from plantar library
                        # ex. box = plantar.BoundBox(...)
                        net_bx[r, c, b, 0:4] = y_Box, x_Box, y_2Box, x_2Box
                        boxes.append(net_bx)  # boxes contain scale

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        constructor
        Write a class Yolo that uses the Yolo v3 algorithm to perform object
            detection:
        class constructor: def __init__(self, model_path, classes_path,
            class_t, nms_t, anchors):
            model_path is the path to where a Darknet Keras model is stored
            classes_path is the path to where the list of class names used
                for the Darknet model, listed in order of index, can be found
            class_t is a float representing the box score threshold for the
                initial filtering step
            nms_t is a float representing the IOU threshold for non-max
                suppression
            anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
                containing all of the anchor boxes:
                outputs is the number of outputs (predictions) made by the
                    Darknet model
                anchor_boxes is the number of anchor boxes used for each
                    prediction
                2 => [anchor_box_width, anchor_box_height]
        Public instance attributes:
            model: the Darknet Keras model
            class_names: a list of the class names for the model
            class_t: the box score threshold for the initial filtering step
            nms_t: the IOU threshold for non-max suppression
            anchors: the anchor boxes
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as classes:
            self.class_names = map(self.validateClass, classes)
        self.classes_path = classes_path
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """sigmoid function"""
        return (1 / (1 + np.exp(-x)))

    def process_outputs(self, outputs, image_size):
        """
        Write a class Yolo (Based on 0-yolo.py):
        Add the public method def process_outputs(self, outputs, image_size):
        outputs is a list of numpy.ndarrays containing the predictions from
        the Darknet model for a single image:
            Each output will have the shape (grid_height, grid_width,
            anchor_boxes, 4 + 1 + classes)
                grid_height & grid_width => the height and width of the grid
                    used for the output
                anchor_boxes => the number of anchor boxes used
                4 => (t_x, t_y, t_w, t_h)
                1 => box_confidence
                classes => class probabilities for all classes
        image_size is a numpy.ndarray containing the image’s original
            size[image_height, image_width]
        Returns a tuple of (boxes, box_confidences, box_class_probs):
            boxes: a list of numpy.ndarrays of
                shape (grid_height, grid_width, anchor_boxes, 4) containing
                the processed boundary boxes for each output, respectively:
                4 => (x1, y1, x2, y2)
                (x1, y1, x2, y2) should represent the boundary box relative
                to original image
            box_confidences: a list of numpy.ndarrays of
                shape (grid_height, grid_width, anchor_boxes, 1)
                containing the box confidences for each output, respectively
            box_class_probs: a list of numpy.ndarrays of
            shape (grid_height, grid_width, anchor_boxes, classes)
            containing the box’s class probabilities
            for each output, respectively
        HINT: The Darknet model is an input to the class for a reason.
        It may not always have the same number of outputs, input sizes, etc.
        """
        boxes = []
        box_class_probs = []
        box_confidences = []
        image_height = image_size[0]
        image_width = image_size[1]
        try:
            input_height = self.model.input.shape[1].value
            input_width = self.model.input.shape[2].value
        except Exception as e:
            input_height = self.model.input.shape[1]
            input_width = self.model.input.shape[2]
        

        for item in range(len(outputs)):
            net_out_pro = outputs[item]
            grid_height, grid_width = net_out_pro.shape[:2]
            net_box = net_out_pro.shape[-2]
            net_class = net_out_pro.shape[-1] - 5
            anchors = self.anchors[item] # load from constructor
            net_out_pro[..., :2] = self.sigmoid(net_out_pro[..., :2])
            net_out_pro[..., 4:] = self.sigmoid(net_out_pro[..., 4:])
            soft_box = net_out_pro[..., :4] # varible soft to be used

            for r in range(grid_height):
                for c in range(grid_width):
                    for b in range(net_box):
                        y, x, w, h, = soft_box[r, c, b, :4]
                        # center image using height and width
                        x = (c + x)
                        ctr_x = x / grid_width
                        y = (r + y)
                        ctr_y = y / grid_height
                        # image height and width
                        w = (anchors[b][0] * np.exp(w))
                        im_width = w / input_width
                        h = (anchors[b][1] * np.exp(h))
                        im_height = h / input_height

                        # define scale
                        x_Box = (ctr_x - im_width/2) * image_width
                        y_Box = (ctr_y - im_height/2) * image_height
                        x_2Box = (ctr_x + im_width/2) * image_width
                        y_2Box = (ctr_y + im_width/2) * image_height

                        # can use BoundBox from plantar library
                        # ex. box = plantar.BoundBox(...)
                        soft_box[r, c, b, 0:4] = y_Box, x_Box, y_2Box, x_2Box
                        boxes.append(soft_box)  # boxes contain scale

            # output confidences
            # box_confidence = [self.sigmoid(net_out_pro[..., 4:5])]
            box_confidence = net_out_pro[..., 4:5]
            # A confidence score of 1 represents 100%
            box_confidences.append(box_confidence)

            # output probabilities
            # ex. If there are 20 classes (C=20),
            # probability, C1, C2, C3,……., C20].
            # box_class_prob = [self.sigmoid(net_out_pro[..., 5:])]
            box_class_prob = net_out_pro[..., 5:]
            box_class_probs.append(box_class_prob)
        # (boxes, box_confidences, box_class_probs)
        return (boxes, box_confidences, box_class_probs)
