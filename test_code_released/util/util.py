from __future__ import print_function
import dlib
import imutils
from imutils.face_utils import rect_to_bb
import torch
import numpy as np
from PIL import Image
import os


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def crop_face(img, size=256, zoomout=1.6):
    """

    Crops all faces in an image.

    Args:
        img: BRG image as an numpy.ndarray.
        size: Width and height to scale resulting crop to.
        zoomout: Zoomout factor.  Scales width and height of region
            around a face.

    Returns:
        A list of 256x256 cropped face BRG images as numpy.ndarrays
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
