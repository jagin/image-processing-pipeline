import os
import json
import numpy as np

from pipeline.pipeline import Pipeline


class SaveSummary(Pipeline):
    """Pipeline task to save processing summary."""

    def __init__(self, filename):
        self.filename = filename

        self.summary = {}
        super(SaveSummary, self).__init__()

    def map(self, data):
        image_id = data["image_id"]
        face_files = data["face_files"]
        faces = data["faces"]

        # Loop over all detected faces and buffer summary results
        self.summary[image_id] = {}
        for i, face in enumerate(faces):
            box, confidence = face
            (x1, y1, x2, y2) = box.astype("int")
            face_file = face_files[i]
            self.summary[image_id][face_file] = {
                "box": np.array([x1, y1, x2, y2], dtype=int).tolist(),
                "confidence": confidence.item()
            }

        return data

    def write(self):
        dirname = os.path.dirname(os.path.abspath(self.filename))
        os.makedirs(dirname, exist_ok=True)

        with open(self.filename, 'w') as json_file:
            json_file.write(json.dumps(self.summary))
