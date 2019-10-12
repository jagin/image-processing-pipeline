from pipeline.pipeline import Pipeline


class DisplaySummary(Pipeline):
    def __init__(self):
        super(DisplaySummary, self).__init__()

    def map(self, data):
        image_file = data["image_file"]
        face_rects = data["face_rects"]

        print(f"[INFO] {image_file}: face detections {len(face_rects)}")

        return data
