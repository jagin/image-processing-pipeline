import cv2

from pipeline.pipeline import Pipeline


class CascadeDetectFaces(Pipeline):
    def __init__(self, classifier):
        # load the face detector
        self.detector = cv2.CascadeClassifier(classifier)

        super(CascadeDetectFaces, self).__init__()

    def map(self, data):
        image = data["image"]

        # Detect faces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_rects = self.detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5,
                                                    minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        data["face_rects"] = face_rects

        return data
