import cv2

from pipeline.pipeline import Pipeline


class CaptureVideo(Pipeline):
    """Pipeline task to capture video stream from file or webcam."""

    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video {src}")

        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if isinstance(src, str) else -1

        super(CaptureVideo, self).__init__()

    def generator(self):
        """Yields the frame content and metadata."""

        frame_idx = 0
        while self.has_next():
            try:
                ret, image = self.cap.read()
                if not ret:
                    # no frames has been grabbed
                    break

                data = {
                    "frame_idx": frame_idx,
                    "image_id": f"{frame_idx:06d}",
                    "image": image,
                }

                if self.filter(data):
                    frame_idx += 1
                    yield self.map(data)
            except StopIteration:
                return

    def cleanup(self):
        """Closes video file or capturing device.

        This function should be triggered after the pipeline completes.
        """

        self.cap.release()
