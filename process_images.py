import cv2
import os
import json
import numpy as np


def parse_args():
    import argparse

    # Parse command line arguments
    ap = argparse.ArgumentParser(description="Image processing pipeline")
    ap.add_argument("-i", "--input", required=True,
                    help="path to input image files")
    ap.add_argument("-o", "--output", default="output",
                    help="path to output directory")
    ap.add_argument("-os", "--out-summary", default=None,
                    help="output JSON summary file name")
    ap.add_argument("-c", "--classifier", default="models/haarcascade/haarcascade_frontalface_default.xml",
                    help="path to where the face cascade resides")

    return ap.parse_args()


def list_images(path, valid_exts=None):
    image_files = []
    # Loop over the input directory structure
    for (root_dir, dir_names, filenames) in os.walk(path):
        for filename in sorted(filenames):
            # Determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()
            if valid_exts and ext.endswith(valid_exts):
                # Construct the path to the file
                file = os.path.join(root_dir, filename)
                image_files.append(file)

    return image_files


def main(args):
    os.makedirs(args.output, exist_ok=True)

    # load the face detector
    detector = cv2.CascadeClassifier(args.classifier)

    # list images from input directory
    input_image_files = list_images(args.input, (".jpg", ".png"))

    # Storage for JSON summary
    summary = {}

    # Loop over the image paths
    for image_file in input_image_files:
        # Load the image and convert it to grayscale
        image = cv2.imread(image_file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        face_rects = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5,
                                               minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        summary[image_file] = {}
        # Loop over all detected faces
        for i, (x, y, w, h) in enumerate(face_rects):
            face = image[y:y+w, x:x+h]

            # Prepare output directory for faces
            output = os.path.join(*(image_file.split(os.path.sep)[1:]))
            output = os.path.join(args.output, output)
            os.makedirs(output, exist_ok=True)

            # Save faces
            face_file = os.path.join(output, f"{i:05d}.jpg")
            cv2.imwrite(face_file, face)

            # Store summary data
            summary[image_file][face_file] = np.array([x, y, w, h], dtype=int).tolist()

        # Display summary
        print(f"[INFO] {image_file}: face detections {len(face_rects)}")

    # Save summary data
    if args.out_summary:
        summary_file = os.path.join(args.output, args.out_summary)
        print(f"[INFO] Saving summary to {summary_file}...")
        with open(summary_file, 'w') as json_file:
            json_file.write(json.dumps(summary))


if __name__ == "__main__":
    args = parse_args()
    main(args)
