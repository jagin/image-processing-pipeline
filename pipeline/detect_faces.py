import numpy as np

from pipeline.pipeline import Pipeline
from pipeline.libs.face_detector import FaceDetector


class DetectFaces(Pipeline):
    """Pipeline task to detect faces from the image"""

    def __init__(self, prototxt, model, batch_size=1, confidence=0.5):
        self.detector = FaceDetector(prototxt, model, confidence=confidence)
        self.batch_size = batch_size

        super(DetectFaces, self).__init__()

    def generator(self):
        """Yields the image enriched with detected faces metadata"""

        batch = []
        stop = False
        while self.has_next() and not stop:
            try:
                # Buffer the pipeline stream
                data = next(self.source)
                batch.append(data)
            except StopIteration:
                stop = True

            # Check if there is anything in batch.
            # Process it if the size match batch_size or there is the end of the input stream.
            if len(batch) and (len(batch) == self.batch_size or stop):
                # Prepare images batch
                images = [data["image"] for data in batch]
                # Detect faces on all images at once
                faces = self.detector.detect(images)

                # Extract the faces metadata and attache them to the proper image
                for image_idx, image_faces in faces.items():
                    batch[image_idx]["faces"] = image_faces

                # Yield all the data from buffer
                for data in batch:
                    if self.filter(data):
                        yield self.map(data)

                batch = []
