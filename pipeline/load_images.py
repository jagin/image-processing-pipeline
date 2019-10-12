import cv2

from pipeline.pipeline import Pipeline
import pipeline.utils as utils


class LoadImages(Pipeline):
    def __init__(self, src, valid_exts=(".jpg", ".png")):
        self.src = src
        self.valid_exts = valid_exts

        super(LoadImages, self).__init__()

    def generator(self):
        source = utils.list_images(self.src, self.valid_exts)
        while self.has_next():
            image_file = next(source)
            image = cv2.imread(image_file)

            data = {
                "image_file": image_file,
                "image": image
            }

            if self.filter(data):
                yield self.map(data)
