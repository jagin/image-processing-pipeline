import cv2

from pipeline.pipeline import Pipeline
import pipeline.libs.utils as utils


class CaptureImages(Pipeline):
    """Pipeline task to capture images from directory"""

    def __init__(self, src, valid_exts=(".jpg", ".png"), level=None):
        self.src = src
        self.valid_exts = valid_exts
        self.level = level

        super(CaptureImages, self).__init__()

    def generator(self):
        """Yields the image content and metadata."""

        source = utils.list_files(self.src, self.valid_exts, self.level)
        while self.has_next():
            try:
                image_file = next(source)
                image = cv2.imread(image_file)

                data = {
                    "image_id": image_file,
                    "image": image
                }

                if self.filter(data):
                    yield self.map(data)
            except StopIteration:
                return
