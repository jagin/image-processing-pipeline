from pipeline.pipeline import Pipeline


class DisplaySummary(Pipeline):
    def __init__(self):
        super(DisplaySummary, self).__init__()

    def map(self, data):
        image_id = data["image_id"]
        face_rects = data["faces"]

        print(f"[INFO] {image_id}: face detections {len(face_rects)}")

        return data
