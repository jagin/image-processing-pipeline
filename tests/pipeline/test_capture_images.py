import numpy as np

from pipeline.capture_images import CaptureImages
import tests.config as config


class TestCaptureImages:
    def test_capture_images(self):
        capture_images = CaptureImages(config.ASSETS_IMAGES_DIR)

        images = list(capture_images)

        assert len(images) == 6
        assert isinstance(images[0]["image"], np.ndarray)
