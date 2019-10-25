import os
import cv2

from pipeline.libs.face_detector import FaceDetector
import tests.config as config


class TestFaceDetector:
    def test_face_detector_with_sample_image(self):
        prototxt = os.path.join(config.MODELS_FACE_DETECTOR_DIR, "deploy.prototxt.txt")
        model = os.path.join(config.MODELS_FACE_DETECTOR_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
        detector = FaceDetector(prototxt, model)

        test_image = cv2.imread(os.path.join(config.ASSETS_IMAGES_DIR, "friends", "friends_01.jpg"))
        faces = detector.detect([test_image])

        assert len(faces) == 1
        assert len(faces[0]) == 3  # Should recognize 3 faces from friends_01.jpg

    def test_face_detector_with_batch_images(self):
        prototxt = os.path.join(config.MODELS_FACE_DETECTOR_DIR, "deploy.prototxt.txt")
        model = os.path.join(config.MODELS_FACE_DETECTOR_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
        detector = FaceDetector(prototxt, model)

        test_image_1 = cv2.imread(os.path.join(config.ASSETS_IMAGES_DIR, "friends", "friends_01.jpg"))
        test_image_2 = cv2.imread(os.path.join(config.ASSETS_IMAGES_DIR, "friends", "friends_02.jpg"))
        test_image_3 = cv2.imread(os.path.join(config.ASSETS_IMAGES_DIR, "friends", "friends_03.jpg"))
        test_image_4 = cv2.imread(os.path.join(config.ASSETS_IMAGES_DIR, "friends", "friends_04.jpg"))
        faces = detector.detect([test_image_1, test_image_2, test_image_3, test_image_4])

        assert len(faces) == 4
        assert len(faces[0])
        assert len(faces[1])
        assert len(faces[2])
        assert len(faces[3])
