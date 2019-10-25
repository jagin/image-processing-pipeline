import cv2
import numpy as np


class FaceDetector:
    def __init__(self, prototxt, model, confidence=0.5):
        self.confidence = confidence

        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)

    def detect(self, images):
        # convert images into blob
        blob = self.preprocess(images)

        # pass the blob through the network and obtain the detections and predictions
        self.net.setInput(blob)
        detections = self.net.forward()
        # Prepare storage for faces for every image in the batch
        faces = dict(zip(range(len(images)), [[] for _ in range(len(images))]))

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence < self.confidence:
                continue

            # grab the image index
            image_idx = int(detections[0, 0, i, 0])
            # grab the image dimensions
            (h, w) = images[image_idx].shape[:2]
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

            # Add result
            faces[image_idx].append((box, confidence))

        return faces

    def preprocess(self, images):
        return cv2.dnn.blobFromImages(images, 1.0, (300, 300), (104.0, 177.0, 123.0))
