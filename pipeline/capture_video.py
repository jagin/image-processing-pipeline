import cv2

from pipeline.pipeline import Pipeline


class CaptureVideo(Pipeline):
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video {src}")

        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        super(CaptureVideo, self).__init__()

    def generator(self):
        while self.has_next():
            ret, image = self.cap.read()
            if not ret:
                break

            data = {
                "image": image
            }
            if self.filter(data):
                yield self.map(data)

    def cleanup(self):
        self.cap.release()
