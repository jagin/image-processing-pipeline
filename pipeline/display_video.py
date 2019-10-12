import cv2

from pipeline.pipeline import Pipeline


class DisplayVideo(Pipeline):
    def __init__(self, window_name='Video'):
        self.window_name = window_name

        cv2.startWindowThread()
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

        super(DisplayVideo, self).__init__()

    def map(self, data):
        image = data["image"]

        cv2.imshow(self.window_name, image)

        # Exit?
        key = cv2.waitKey(1) & 0xFF
        # Esc key pressed or window closed?
        if key == 27 or cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            raise StopIteration

        return data

    def cleanup(self):
        cv2.destroyWindow(self.window_name)
