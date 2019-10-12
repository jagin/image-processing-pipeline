import cv2

from pipeline.pipeline import Pipeline


class SaveVideo(Pipeline):
    def __init__(self, filename, fps=30, fourcc='MJPG'):
        self.filename = filename
        self.fps = fps
        self.writer = None
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)

        super(SaveVideo, self).__init__()

    def map(self, data):
        image = data["image"]
        h, w = image.shape[:2]

        if self.writer is None:
            self.writer = cv2.VideoWriter(self.filename, self.fourcc, self.fps, (w, h), image.ndim == 3)
        self.writer.write(image)

        return data

    def cleanup(self):
        if self.writer:
            self.writer.release()
