import os


def list_images(path, valid_exts=None):
    # Loop over the input directory structure
    for (root_dir, dir_names, filenames) in os.walk(path):
        for filename in sorted(filenames):
            # Determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()
            if valid_exts and ext.endswith(valid_exts):
                # Construct the path to the file and yield it
                file = os.path.join(root_dir, filename)
                yield file
