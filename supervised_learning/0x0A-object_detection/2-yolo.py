#!/usr/bin/env python3
""" class Yolo that uses the Yolo v3 algorithm to perform object detection """
import tensorflow.keras as K
import tensorflow as tf
import numpy as np


def sigmoid(x):
    """ Sigmoid function """
    return 1/(1 + np.exp(-x))


class Yolo:
    """ YOLO Class """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """ YOLO Class """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = f.read().splitlines()
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """ YOLO Class """
        boxes = []
        box_confidences = []
        box_class_probs = []
        height, width = image_size
        i = 0

        for output in outputs:
            grid_height = output.shape[0]
            grid_width = output.shape[1]
            anchors = self.anchors[i]
            box = output[:, :, :, :4]
            for cx in range(len(output)):
                for cy in range(len(output[cx])):
                    for anchor in range(len(output[cx][cy])):
                        pw, ph = anchors[anchor]
                        tx = output[cx, cy, anchor, 0]
                        ty = output[cx, cy, anchor, 1]
                        tw = output[cx, cy, anchor, 2]
                        th = output[cx, cy, anchor, 3]
                        bx = sigmoid(tx) + cy
                        bx = bx / grid_width
                        by = sigmoid(ty) + cx
                        by = by / grid_height
                        bw = pw * np.exp(tw)
                        bw = bw / self.model.input.shape[1].value
                        bh = ph * np.exp(th)
                        bh = bh / self.model.input.shape[2].value
                        x1 = (bx - bw / 2)
                        x2 = (x1 + bw)
                        y1 = (by - bh / 2)
                        y2 = (y1 + bh)
                        box[cx, cy, anchor, 0] = x1 * width
                        box[cx, cy, anchor, 1] = y1 * height
                        box[cx, cy, anchor, 2] = x2 * width
                        box[cx, cy, anchor, 3] = y2 * height
            i += 1
            boxes.append(box)
            box_confidences.append(sigmoid(output[:, :, :, 4:5]))
            box_class_probs.append(sigmoid(output[:, :, :, 5:]))
        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """ YOLO Class """
        filtered_scores = []
        filtered_boxes = []
        filtered_classes = []
        for i in range(len(boxes)):
            box_scores = box_confidences[i] * box_class_probs[i]
            box_classes = np.argmax(box_scores, axis=-1)
            box_scores = np.max(box_scores, axis=-1)
            filtering_mask = box_scores >= self.class_t

            filtered_scores += box_scores[filtering_mask].tolist()
            filtered_boxes += boxes[i][filtering_mask].tolist()
            filtered_classes += box_classes[filtering_mask].tolist()
        filtered_boxes = np.array(filtered_boxes)
        filtered_classes = np.array(filtered_classes)
        filtered_scores = np.array(filtered_scores)
        return filtered_boxes, filtered_classes, filtered_scores
