import os

MAIN_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")

ASSETS_DIR = os.path.join(MAIN_DIR, "assets")
ASSETS_IMAGES_DIR = os.path.join(ASSETS_DIR, "images")
ASSETS_VIDEOS_DIR = os.path.join(ASSETS_DIR, "videos")

MODELS_DIR = os.path.join(MAIN_DIR, "models")
MODELS_FACE_DETECTOR_DIR = os.path.join(MODELS_DIR, "face_detector")
