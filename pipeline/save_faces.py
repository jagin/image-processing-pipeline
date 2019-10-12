import os
import cv2

from pipeline.pipeline import Pipeline


class SaveFaces(Pipeline):
    def __init__(self, path):
        self.path = path

        super(SaveFaces, self).__init__()

    def map(self, data):
        image_file = data["image_file"]
        image = data["image"]
        face_rects = data["face_rects"]
        data["face_files"] = []

        # Loop over all detected faces
        for i, (x, y, w, h) in enumerate(face_rects):
            face = image[y:y+w, x:x+h]

            # Prepare output directory for faces
            output = os.path.join(*(image_file.split(os.path.sep)[1:]))
            output = os.path.join(self.path, output)
            os.makedirs(output, exist_ok=True)

            # Save faces
            face_file = os.path.join(output, f"{i:05d}.jpg")
            data["face_files"].append(face_file)
            cv2.imwrite(face_file, face)

        return data
