import os
from tqdm import tqdm

from pipeline.capture_video import CaptureVideo
from pipeline.detect_faces import DetectFaces
from pipeline.save_faces import SaveFaces
from pipeline.save_summary import SaveSummary
from pipeline.annotate_image import AnnotateImage
from pipeline.display_video import DisplayVideo
from pipeline.save_video import SaveVideo


def parse_args():
    import argparse

    # Parse command line arguments
    ap = argparse.ArgumentParser(description="Video processing pipeline")
    ap.add_argument("-i", "--input", default="0",
                    help="path to input video file or camera identifier")
    ap.add_argument("-o", "--output", default="output",
                    help="path to output directory")
    ap.add_argument("-ov", "--out-video", default=None,
                    help="output video file name")
    ap.add_argument("-os", "--out-summary", default="summary.json",
                    help="output JSON summary file name")
    ap.add_argument("-p", "--progress", action="store_true", help="display progress")
    ap.add_argument("-d", "--display", action="store_true", help="display video result")
    ap.add_argument("--prototxt", default="./models/face_detector/deploy.prototxt.txt",
                    help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("--model", default="./models/face_detector/res10_300x300_ssd_iter_140000.caffemodel",
                    help="path to Caffe pre-trained model")
    ap.add_argument("--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak face detections")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="face detection batch size")

    return ap.parse_args()


def main(args):
    # Create pipeline steps
    capture_video = CaptureVideo(int(args.input) if args.input.isdigit() else args.input)

    detect_faces = DetectFaces(prototxt=args.prototxt, model=args.model,
                               confidence=args.confidence, batch_size=args.batch_size)

    save_faces = SaveFaces(args.output)

    summary_file = os.path.join(args.output, args.out_summary)
    save_summary = SaveSummary(summary_file)

    annotate_image = AnnotateImage("annotated_image") \
        if args.display or args.out_video else None

    display_video = DisplayVideo("annotated_image") \
        if args.display else None

    save_video = SaveVideo("annotated_image", os.path.join(args.output, args.out_video), fps=capture_video.fps) \
        if args.out_video else None

    # Create image processing pipeline
    pipeline = (capture_video |
                detect_faces |
                save_faces |
                annotate_image |
                display_video |
                save_video |
                save_summary)

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
        progress.close()

        # Pipeline cleanup
        capture_video.cleanup()
        if display_video:
            display_video.cleanup()
        if save_video:
            save_video.cleanup()

        print(f"[INFO] Saving summary to {summary_file}...")
        save_summary.write()


if __name__ == "__main__":
    args = parse_args()
    main(args)
