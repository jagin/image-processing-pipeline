import os
import json
import numpy as np

from pipeline.pipeline import Pipeline


class SaveSummary(Pipeline):
    def __init__(self, filename):
        self.filename = filename

        self.summary = {}
        super(SaveSummary, self).__init__()

    def map(self, data):
        image_file = data["image_file"]
        face_files = data["face_files"]
        face_rects = data["face_rects"]

        # Buffer summary results
        self.summary[image_file] = {}
        for i, (x, y, w, h) in enumerate(face_rects):
            face_file = face_files[i]
            self.summary[image_file][face_file] = np.array([x, y, w, h], dtype=int).tolist()

        return data

    def write(self):
        dirname = os.path.dirname(os.path.abspath(self.filename))
        os.makedirs(dirname, exist_ok=True)

        with open(self.filename, 'w') as json_file:
            json_file.write(json.dumps(self.summary))
