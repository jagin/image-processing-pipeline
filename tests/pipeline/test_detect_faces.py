import os

from pipeline.capture_images import CaptureImages
from pipeline.detect_faces import DetectFaces
import tests.config as config


class TestDetectFaces:
    def test_detect_faces(self):
        capture_images = CaptureImages(os.path.join(config.ASSETS_IMAGES_DIR, "friends"))
        prototxt = os.path.join(config.MODELS_FACE_DETECTOR_DIR, "deploy.prototxt.txt")
        model = os.path.join(config.MODELS_FACE_DETECTOR_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
        detect_faces = DetectFaces(prototxt=prototxt, model=model)

        pipeline = iter(capture_images |
                        detect_faces)

        data = next(pipeline)
        assert len(data["faces"])

        data = next(pipeline)
        assert len(data["faces"])

        data = next(pipeline)
        assert len(data["faces"])

        data = next(pipeline)
        assert len(data["faces"])

    def test_detect_faces_in_batch(self):
        capture_images = CaptureImages(os.path.join(config.ASSETS_IMAGES_DIR, "friends"))
        prototxt = os.path.join(config.MODELS_FACE_DETECTOR_DIR, "deploy.prototxt.txt")
        model = os.path.join(config.MODELS_FACE_DETECTOR_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
        detect_faces = DetectFaces(prototxt=prototxt, model=model, batch_size=2)

        pipeline = iter(capture_images |
                        detect_faces)

        data = next(pipeline)
        assert len(data["faces"])

        data = next(pipeline)
        assert len(data["faces"])

        data = next(pipeline)
        assert len(data["faces"])

        data = next(pipeline)
        assert len(data["faces"])
