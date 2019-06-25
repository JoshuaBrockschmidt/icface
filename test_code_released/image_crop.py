#!/usr/bin/env python3

import argparse
import cv2
import numpy as np
import os
from PIL import Image
import re
from sys import stderr
from util.util import crop_face

def get_next_save_path(path):
    """
    Returns the next viable save path within a directory.
    For instance, if the files "1.png" and "2.png" are present,
    it will return "3.png".

    Args:
        path: Directory to save within.

    Returns:
        Next viable save path.

    """
    files = os.listdir(path)
    largest = 0
    for f in files:
        # Only look at PNG files with numbered names.
        if re.search(r'[0-9]+\.png$', f):
            num = int(os.path.splitext(f)[0])
            if num > largest:
                largest = num
    save_path = os.path.abspath('{}/{}.png'.format(path, largest + 1))
    return save_path

def process_image(path):
    """
    Find all faces in an image, crops them, and saves them in new_crop.

    Args:
        path: Path to image to process.

    Return:
        Number of faces cropped on success.  None on failure.
    """
    print('Cropping "{}"...'.format(os.path.abspath(path)))
    if not os.path.exists(path):
        print('File "{}" does not exist'.format(path), file=stderr)
        return None
    if not os.path.isfile(path):
        print('File "{}" is not a file'.format(path), file=stderr)
        return None

    # Attempt to read image.
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print('Failed to read "{}"'.format(path), file=stderr)
        return None

    faces = crop_face(img)
    if faces is None:
        print('Failed to find a face in "{}"'.format(path), file=stderr)

    num_cropped = 0
    for face in faces:
        try:
            save_path = get_next_save_path('./new_crop')
            print('Writing to "{}"...'.format(save_path))
            cv2.imwrite(save_path, face)
            num_cropped += 1
        except KeyboardInterrupt as e:
            # Safely handle premature termination.
            # Remove unfinished file.
            if os.exists(save_path):
                os.remove(save_path)
            raise e

    return num_cropped

def main(paths):
    total_cropped = 0
    for path in paths:
        cropped = process_image(path)
        if not total_cropped is None:
            total_cropped += cropped
    print('{} faces cropped'.format(total_cropped))

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(
            description='Crops an image around the face')
        parser.add_argument('paths', metavar='image',
                            type=str, nargs='+',
                            help='Path to image')
        args = parser.parse_args()

        main(args.paths)
    except KeyboardInterrupt:
        print('Program terminated')
