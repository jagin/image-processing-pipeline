import os
import cv2

from pipeline.pipeline import Pipeline


class SaveFaces(Pipeline):
    """Pipeline task to save detected faces."""

    def __init__(self, path, image_ext="jpg"):
        self.path = path
        self.image_ext = image_ext

        super(SaveFaces, self).__init__()

    def map(self, data):
        image_id = data["image_id"]
        image = data["image"]
        faces = data["faces"]
        data["face_files"] = []

        # Loop over all detected faces
        for i, face in enumerate(faces):
            box, confidence = face
            (x1, y1, x2, y2) = box.astype("int")
            # Crop the face from the image
            face = image[y1:y2, x1:x2]

            # Prepare output directory for faces
            output = os.path.join(*(image_id.split(os.path.sep)))
            output = os.path.join(self.path, output)
            os.makedirs(output, exist_ok=True)

            # Save faces
            face_file = os.path.join(output, f"{i:05d}.{self.image_ext}")
            data["face_files"].append(face_file)
            cv2.imwrite(face_file, face)

        return data
