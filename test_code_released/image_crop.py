#!/usr/bin/env python3

import argparse
#from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib
import cv2
#from imutils import face_utils
import numpy as np
import os
from PIL import Image
#import pdb
from sys import stderr

def crop_face(img, size=256):
    """Crops all faces in an image.

    Args:
        img: Image to crop, as an numpy.ndarray.
        size: Width and height to scale resulting crop to.
    Returns:
        A list of 256x256 cropped face images as a numpy.ndarray
        if at least one face is found.
        None if no faces are found.
    """
    detector = dlib.get_frontal_face_detector()

    image= cv2.resize(img, (400, 400))

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   
    rects = detector(rgb, 1)
    faces = dlib.full_object_detections()

    if len(rects) == 0:
        # No faces were detected.
        return None
    else:
        # Crop all detected faces.
        cropped_faces = []
        for rect in rects:
            c1=rect.dcenter()
            (x, y, w, h) = rect_to_bb(rect)
            w=np.int(w*1.6) 
            h=np.int(h*1.6) 
            x=c1.x-np.int(w/2.0)
            y=c1.y-np.int(h/2.0)
            if y<0:
                y=0
            if x<0:
                x=0
                

            faceOrig = imutils.resize(rgb[y:y+h, x:x+w],height=size) #y=10,h+60,W+40

            d_num = np.asarray(faceOrig)
            cropped_faces.append(d_num)

        return cropped_faces

def main(path):
    if not os.path.exists(path):
        print('File "{}" does not exist'.format(path), file=stderr)
        exit(1)
    if not os.path.isfile(path):
        print('File "{}" is not a file'.format(path), file=stderr)
        exit(1)

    # Attempt to read image.
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print('Failed to read "{}"'.format(path), file=stderr)
        exit(1)

    faces = crop_face(img)
    if faces is None:
        print('Failed to find a face in "{}"'.format(path), file=stderr)

    for face in faces:
        c = 1240  # Temporary
        save_path = './new_crop/' + str(c) + '.png'
        f_im = Image.fromarray(face)
        f_im.save(save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Crops an image around the face')
    parser.add_argument('path', metavar='image',
                        type=str, nargs=1,
                        help='Path to image')
    args = parser.parse_args()

    main(args.path[0])
