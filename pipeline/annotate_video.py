import cv2

from pipeline.pipeline import Pipeline

class AnnotateVideo(Pipeline):
    def __init__(self):
        super(AnnotateVideo, self).__init__()

    def map(self, data):
        image = data["image"]
        face_rects = data["face_rects"]

        # loop over the faces and draw a rectangle around each
        for (x, y, w, h) in face_rects:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return data
