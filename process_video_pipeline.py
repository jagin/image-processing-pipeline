import os
from tqdm import tqdm

from pipeline.capture_video import CaptureVideo
from pipeline.cascade_detect_faces import CascadeDetectFaces
from pipeline.annotate_video import AnnotateVideo
from pipeline.display_video import DisplayVideo
from pipeline.save_video import SaveVideo


def parse_args():
    import argparse

    # Parse command line arguments
    ap = argparse.ArgumentParser(description="Video processing pipeline")
    ap.add_argument("-i", "--input", default=0,
                    help="path to input video file or camera identifier")
    ap.add_argument("-o", "--output", default="output",
                    help="path to output directory")
    ap.add_argument("-ov", "--out-video", default=None,
                    help="output video file name")
    ap.add_argument("-c", "--classifier", default="models/haarcascade/haarcascade_frontalface_default.xml",
                    help="path to where the face cascade resides")
    ap.add_argument("-p", "--progress", action="store_true", help="display progress")

    return ap.parse_args()


def main(args):
    # Create pipeline steps
    capture_video = CaptureVideo(args.input)

    detect_faces = CascadeDetectFaces(args.classifier)

    annotate_video = AnnotateVideo()

    display_video = DisplayVideo()

    if args.out_video:
        video_file = os.path.join(args.output, args.out_video)
        save_video = SaveVideo(video_file, fps=capture_video.fps)

    # Create image processing pipeline
    pipeline = capture_video | detect_faces | annotate_video | display_video
    if args.out_video:
        pipeline |= save_video

    # Iterate through pipeline
    progress = tqdm(total=capture_video.frame_count if capture_video.frame_count > 0 else None,
                    disable=not args.progress)
    try:
        for _ in pipeline:
            progress.update(1)
    except StopIteration:
        return
    except KeyboardInterrupt:
        return
    finally:
        capture_video.cleanup()
        display_video.cleanup()
        if args.out_video:
            save_video.cleanup()


if __name__ == "__main__":
    args = parse_args()
    main(args)
