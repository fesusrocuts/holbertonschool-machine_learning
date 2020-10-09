#!/usr/bin/env python3
"""
0x0A. Object Detection with Yolo
"""
import tensorflow.keras as K
import numpy as np
import cv2
import glob
import os


class Yolo():
    """class Yolo"""

    def validateClass(self, listC):
        """function used to remove spaces"""
        arr = []
        for item in listC:
            arr.append(item.strip(" "))
        return arr

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
            input_height = int(self.model.input.shape[1])
            input_width = int(self.model.input.shape[2])

        for item in range(len(outputs)):
            net_out_pro = outputs[item]
            grid_height, grid_width = net_out_pro.shape[:2]
            net_box = net_out_pro.shape[-2]
            net_class = net_out_pro.shape[-1] - 5
            # load from constructor
            anchors = self.anchors[item]
            net_out_pro[..., :2] = self.sigmoid(net_out_pro[..., :2])
            net_out_pro[..., 4:] = self.sigmoid(net_out_pro[..., 4:])
            # varible soft to be used
            soft_box = net_out_pro[..., :4]

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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Write a class Yolo (Based on 1-yolo.py):
        Add the public method def filter_boxes(self, boxes,
        box_confidences, box_class_probs):
        boxes: a list of numpy.ndarrays of shape (grid_height,
        grid_width, anchor_boxes, 4) containing the processed
        boundary boxes for each output, respectively
        box_confidences: a list of numpy.ndarrays of
        shape (grid_height, grid_width, anchor_boxes, 1)
        containing the processed box confidences for each output,
        respectively
        box_class_probs: a list of numpy.ndarrays of
        shape (grid_height, grid_width, anchor_boxes, classes)
        containing the processed box class probabilities
        for each output, respectively
        Returns a tuple of (filtered_boxes, box_classes,
        box_scores):
            filtered_boxes: a numpy.ndarray of
            shape (?, 4) containing all of the filtered
            bounding boxes:
            box_classes: a numpy.ndarray of
            shape (?,) containing the class number that
            each box in filtered_boxes predicts, respectively
            box_scores: a numpy.ndarray of shape (?)
            containing the box scores for each box in
            filtered_boxes, respectively
        """
        new_boxes = np.concatenate([boxs.reshape(-1, 4) for boxs in boxes])
        class_probs = np.concatenate([probs.reshape(-1,
                                                    box_class_probs[0].
                                                    shape[-1])
                                      for probs in box_class_probs])
        all_classes = class_probs.argmax(axis=1)
        all_confidences = (np.concatenate([conf.reshape(-1)
                                           for conf in box_confidences])
                           * class_probs.max(axis=1))
        trash_idx = np.where(all_confidences < self.class_t)
        return (np.delete(new_boxes, trash_idx, axis=0),
                np.delete(all_classes, trash_idx),
                np.delete(all_confidences, trash_idx))

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Write a class Yolo (Based on 2-yolo.py):
        Add the public method
        def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        filtered_boxes: a numpy.ndarray of shape (?, 4) containing all of the
        filtered bounding boxes:
        box_classes: a numpy.ndarray of shape (?,) containing the class number
        for the class that filtered_boxes predicts, respectively
        box_scores: a numpy.ndarray of shape (?) containing the box scores for
        each box in filtered_boxes, respectively
        Returns a tuple of
        (box_predictions, predicted_box_classes, predicted_box_scores):
            box_predictions: a numpy.ndarray of shape (?, 4) containing all of
            the predicted bounding boxes ordered by class and box score
            predicted_box_classes: a numpy.ndarray of shape (?,) containing the
            class number for box_predictions ordered by class and box score,
            respectively predicted_box_scores: a numpy.ndarray of shape (?)
            containing the box scores for box_predictions ordered by class and
            box score, respectively
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for box_class in set(box_classes):
            # where they are the same
            idx = np.where(box_classes == box_class)

            # function arrays
            fb_i = filtered_boxes[idx]
            bc_i = box_classes[idx]
            bS_i = box_scores[idx]

            # coordinates of the bounding boxes
            x1 = fb_i[:, 0]
            y1 = fb_i[:, 1]
            x2 = fb_i[:, 2]
            y2 = fb_i[:, 3]

            # calculate area of the bounding boxes and sort
            union_area = (x2 - x1 + 1) * (y2 - y1 + 1)
            sort_order = bS_i.argsort()[::-1]

            # loop remaining indexes
            pkd_idxs = []  # to hold list of picked indexes
            while len(sort_order) > 0:
                i_pos = sort_order[0]
                l_pos = sort_order[1:]
                pkd_idxs.append(i_pos)

            # find the coordinates of intersection
                xx1 = np.maximum(x1[i_pos], x1[l_pos])
                yy1 = np.maximum(y1[i_pos], y1[l_pos])
                xx2 = np.minimum(x2[i_pos], x2[l_pos])
                yy2 = np.minimum(y2[i_pos], y2[l_pos])

            # width and height of bounding box
            wb = np.maximum(0, xx2 - xx1 + 1)
            hb = np.maximum(0, yy2 - yy1 + 1)

            # overlap ratio betw bounding box
            interSect = (hb * wb)
            overlap = union_area[i_pos] + union_area[l_pos] - interSect
            iou = interSect / overlap

            # below Threshold
            # nms_t is a float representing the IOU threshold for non-max
            #    suppression
            below_Thresh = np.where(iou <= self.nms_t)[0]
            sort_order = sort_order[below_Thresh + 1]

            pkd = np.array(pkd_idxs)  # array of piked indexes

            # append picked to function arrays
            box_predictions.append(pkd)
            predicted_box_classes.append(pkd)
            predicted_box_scores.append(pkd)

        box_predictions = np.concatenate(box_predictions)
        predicted_box_classes = np.concatenate(predicted_box_classes)
        predicted_box_scores = np.concatenate(predicted_box_scores)

        return (box_predictions, predicted_box_classes, predicted_box_scores)

    @staticmethod
    def load_images(folder_path):
        """
        Write a class Yolo (Based on 3-yolo.py):
        Add the static method def load_images(folder_path):
        folder_path: a string representing the path to the folder
        holding all the images to load
        Returns a tuple of (images, image_paths):
            images: a list of images as numpy.ndarrays
            image_paths: a list of paths to the individual
            images in images
        """
        images = []
        image_paths = glob.glob(folder_path + "/*")

        for item in image_paths:
            image = cv2.imread(item)
            images.append(image)

        return (images, image_paths)

    def preprocess_images(self, images):
        """
        Write a class Yolo (Based on 4-yolo.py):
        Add the public method def preprocess_images(self, images):
        images: a list of images as numpy.ndarrays
        Resize the images with inter-cubic interpolation
        Rescale all images to have pixel values in the range [0, 1]
        Returns a tuple of (pimages, image_shapes):
            pimages: a numpy.ndarray of shape (ni, input_h, input_w, 3)
            containing all of the preprocessed images
                ni: the number of images that were preprocessed
                input_h: the input height for the Darknet model
                Note: this can vary by model
                input_w: the input width for the Darknet model
                Note: this can vary by model
                3: number of color channels
            image_shapes: a numpy.ndarray of shape (ni, 2) containing
            the original height and width of the images
                2 => (image_height, image_width)
        """
        try:
            input_height = self.model.input.shape[1].value
            input_width = self.model.input.shape[2].value
        except Exception as e:
            input_height = int(self.model.input.shape[1])
            input_width = int(self.model.input.shape[2])
        nSize_images = []
        im_shape = []

        for item in images:
            image = cv2.resize(item, (input_width, input_height),
                               interpolation=cv2.INTER_CUBIC)
            image = image.astype(float) / 255  # colors
            nSize_images.append(image)
        pimages = np.stack(nSize_images, axis=0)
        # apply shape to images
        im_shape = [item.shape[:2] for item in images]
        image_shapes = np.stack(im_shape, axis=0)
        return (pimages, image_shapes)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Write a class Yolo (Based on 5-yolo.py):
        Add the public method def show_boxes(self, image, boxes,
        box_classes, box_scores, file_name):
        image: a numpy.ndarray containing an unprocessed image
        boxes: a numpy.ndarray containing the boundary boxes
        for the image box_classes: a numpy.ndarray containing
        the class indices for each box
        box_scores: a numpy.ndarray containing the box scores
        for each box file_name: the file path where the original
        image is stored
        Displays the image with all boundary boxes, class names,
        and box scores (see example below)
            Boxes should be drawn as with a blue line of
            thickness 2
            Class names and box scores should be drawn above
            each box in red
                Box scores should be rounded to 2 decimal places
                Text should be written 5 pixels above the top left
                corner of the box
                Text should be written in FONT_HERSHEY_SIMPLEX
                Font scale should be 0.5
                Line thickness should be 1
                You should use LINE_AA as the line type
            The window name should be the same as file_name
            If the s key is pressed:
                The image should be saved in the directory detections,
                located in the current directory
                If detections does not exist, create it
                The saved image should have the file name file_name
                The image window should be closed
            If any key besides s is pressed, the image window should
            be closed without saving
        """
        original_image = image
        for idx, box in enumerate(boxes):
            # bounding boxes
            start_x = int(box[0])
            start_y = int(box[1])
            end_x = int(box[2])
            end_y = int(box[3])

            # color
            textColor = (0, 0, 255)
            lineColor = (255, 0, 0)

            # box
            original_image = cv2.rectangle(original_image,
                                       (start_x, start_y), (end_x, end_y),
                                       lineColor, thickness=2)
            # text
            original_image = cv2.putText(original_image,
                                     self.class_names[box_classes[idx]]
                                     + " " + "{:.2f}".format(box_scores[idx]),
                                     (start_x, (end_y) - 5),
                                     cv2.FONT_HERSHEY_SIMPLEX,
                                     0.5, textColor,
                                     thickness=1,
                                     lineType=cv2.LINE_AA,
                                     bottomLeftOrigin=False)
            cv2.imshow(file_name, image)
            key = cv2.waitKey(0)
            if key == ('s'):
                if not os.path.exists('detections'):
                    os.makedirs('detections')
                os.chdir('detections')
                cv2.imwrite(file_name, image)
                os.chdir('../')
            cv2.destroyAllWindows()

    def predict(self, folder_path):
        """
        Write a class Yolo (Based on 6-yolo.py):
        Add the public method def predict(self, folder_path):
        folder_path: a string representing the path to the
        folder holding all the images to predict
        All image windows should be named after the corresponding
        image filename without its full path(see examples below)
        Displays all images using the show_boxes method
        Returns: a tuple of (predictions, image_paths):
            predictions: a list of tuples for each image
            of (boxes, box_classes, box_scores)
            image_paths: a list of image paths corresponding to
            each prediction in predictions
        """
        images, image_paths = Yolo.load_images(folder_path)
        pimages, image_shapes = self.preprocess_images(images)
        outputs = self.model.predict(pimages)
        predictions = []
        for i in range(pimages.shape[0]):
            for out_i in outputs:
                boxes, box_conf, box_cl_probs = (self.process_outputs(out_i[i],
                                                image_shapes[i]))
                filtr_boxes, bx_class, bx_scores = (self.filter_boxes(boxes,
                                                   box_conf, box_cl_probs))
                pred_boxs, pred_bcl, pred_bsc = (self.
                                                non_max_suppression(filtr_boxes,
                                                                    bx_class,
                                                                    bx_scores))

        predictions.append((pred_boxs, pred_bcl, pred_bsc))
        file_name = image_paths[i].split("/")[-1]
        self.show_boxes(images[i], pred_boxs, pred_bcl, pred_bsc, file_name)

        return (predictions, image_paths)
