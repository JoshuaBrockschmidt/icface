#!/usr/bin/env python3

import argparse
from imutils.face_utils import rect_to_bb
import imutils
import dlib
import cv2
import numpy as np
import os
from PIL import Image
import re
from sys import stderr

def crop_face(img, size=256, zoomout=1.6):
    """Crops all faces in an image.

    Args:
        img: Image to crop, as an numpy.ndarray.
        size: Width and height to scale resulting crop to.
        zoomout: Zoomout factor.  Scales width and height of region
            around a face.
    Returns:
        A list of 256x256 cropped face images as a numpy.ndarray
        if at least one face is found.  None if no faces are found.
    """
    detector = dlib.get_frontal_face_detector()

    rects = detector(img, 1)
    faces = dlib.full_object_detections()

    if len(rects) == 0:
        # No faces were detected.
        return None
    else:
        # Crop all detected faces.
        cropped_faces = []
        for rect in rects:
            c1 = rect.dcenter()
            (x, y, w, h) = rect_to_bb(rect)
            w = np.int(w * zoomout)
            h = np.int(h * zoomout)
            x = c1.x - np.int(w / 2.0)
            y = c1.y - np.int(h / 2.0)
            if y < 0:
                y = 0
            if x < 0:
                x = 0

            face_orig = imutils.resize(img[y:y+h, x:x+w], height=size)  #y=10,h+60,W+40
            cropped_faces.append(face_orig)

        return cropped_faces

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
    """Find all faces in an image, crops them, and saves them in new_crop.

    Args:
        path: Path to image to process.
    Return:
        True on success, False on failure.
    """
    if not os.path.exists(path):
        print('File "{}" does not exist'.format(path), file=stderr)
        return False
    if not os.path.isfile(path):
        print('File "{}" is not a file'.format(path), file=stderr)
        return False

    # Attempt to read image.
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print('Failed to read "{}"'.format(path), file=stderr)
        return False

    faces = crop_face(img)
    if faces is None:
        print('Failed to find a face in "{}"'.format(path), file=stderr)

    for face in faces:
        save_path = get_next_save_path('./new_crop')
        cv2.imwrite(save_path, face)

    return True

def main(paths):
    for path in paths:
        process_image(path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Crops an image around the face')
    parser.add_argument('paths', metavar='image',
                        type=str, nargs='+',
                        help='Path to image')
    args = parser.parse_args()

    main(args.paths)
