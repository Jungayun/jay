#!/usr/bin/python

# Copyright (c) 2015 Matthew Earl
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
This is the code behind the Switching Eds blog post:

    http://matthewearl.github.io/2015/07/28/switching-eds-with-python/

See the above for an explanation of the code below.

To run the script you'll need to install dlib (http://dlib.net) including its
Python bindings, and OpenCV. You'll also need to obtain the trained model from
sourceforge:

    http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2

Unzip with `bunzip2` and change `PREDICTOR_PATH` to refer to this file. The
script is run like so:

    ./faceswap.py <head image> <face image>

If successful, a file `output.jpg` will be produced with the facial features
from `<head image>` replaced with the facial features from `<face image>`.

"""

import cv2
import dlib
import numpy
import tensorflow
import sys

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1
FEATHER_AMOUNT = 11
FEATHER_AMOUNT2 = int(11/3)

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(49, 61))
MOUTH_POINTS_UP = list(range(49, 51))
MOUTH_POINTS_UP2 = list(range(51, 54))
MOUTH_POINTS_UP_ = list(range(61, 62))
MOUTH_POINTS_UP__ = list(range(62, 63))
MOUTH_POINTS_DOWN = list(range(55, 58))
MOUTH_POINTS_DOWN2 = list(range(58, 60))
MOUTH_POINTS_DOWN_ = list(range(65, 66))
MOUTH_POINTS_DOWN__ = list(range(66, 68))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                               RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS_UP_ + MOUTH_POINTS_UP__ + MOUTH_POINTS_UP + MOUTH_POINTS_UP2+ MOUTH_POINTS_DOWN + MOUTH_POINTS_UP2 + MOUTH_POINTS_DOWN_ + MOUTH_POINTS_DOWN__)

# ALIGN_POINTS2 = (MOUTH_POINTS)
# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
OVERLAY_POINTS = [
    LEFT_EYE_POINTS , RIGHT_EYE_POINTS , LEFT_BROW_POINTS , RIGHT_BROW_POINTS,
    NOSE_POINTS , MOUTH_POINTS_UP_ , MOUTH_POINTS_UP__ , MOUTH_POINTS_UP , MOUTH_POINTS_UP2, MOUTH_POINTS_DOWN , MOUTH_POINTS_DOWN2 , MOUTH_POINTS_DOWN_ , MOUTH_POINTS_DOWN__
]
#
# OVERLAY_POINTS = [
#     LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
#     NOSE_POINTS]

# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.6

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

#class TooManyFaces(Exception):
#    pass

class NoFaces(Exception):
    pass

def get_landmarks(im):
    rects = detector(im, 1)
    faces_list = []
    faces_num = len(rects)

    if len(rects) == 0:
        faces_num = 0
        return faces_num, []
        # raise NoFaces

    for i in range(faces_num):
        result = numpy.matrix([[p.x, p.y] for p in predictor(im, rects[i]).parts()])
        # result = numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
        faces_list.append(result)

    # if len(rects) > 1:
    #     raise TooManyFaces

    # return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
    return faces_num, faces_list[0]


def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)
#
# def get_face_mask(im, landmarks):
#     im = numpy.zeros(im.shape[:2], dtype=numpy.float64)
#
#     for group in OVERLAY_POINTS:
#         draw_convex_hull(im,
#                          landmarks[group],
#                          color=1)
#
#     im = numpy.array([im, im, im]).transpose((1, 2, 0))
#
#     im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
#     im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)
#
#     return im
def get_face_mask(im, landmarks):
    im = numpy.zeros(im.shape[:2], dtype=numpy.float64)
    im1 = numpy.zeros(im.shape[:2], dtype=numpy.float64)

    for group in OVERLAY_POINTS[:5]:
        draw_convex_hull(im, landmarks[group], color=1)

    im = numpy.array([im, im, im]).transpose((1, 2, 0))
    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    for group in OVERLAY_POINTS[5:]:
        draw_convex_hull(im1, landmarks[group], color=1)

    # FEATHER_AMOUNT = int(FEATHER_AMOUNT/2)
    im1 = numpy.array([im1, im1, im1]).transpose((1, 2, 0))
    im1 = (cv2.GaussianBlur(im1, (FEATHER_AMOUNT2, FEATHER_AMOUNT2), 0) > 0) * 1.0
    im1 = cv2.GaussianBlur(im1, (FEATHER_AMOUNT2, FEATHER_AMOUNT2), 0)

    return im+im1

def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:

        sum ||s*R*p1,i + T - p2,i||^2

    is minimized.

    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    c1 = numpy.mean(points1, axis=0)


    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = numpy.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])

def read_im_and_landmarks(im):
    # im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    faces_num, faces_list = get_landmarks(im)

    return im, faces_list, faces_num

def warp_im(im, M, dshape):
    output_im = numpy.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
                              numpy.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              numpy.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) /
                                                im2_blur.astype(numpy.float64))

camera = cv2.VideoCapture(0)
im = cv2.imread('heart3.png', cv2.IMREAD_COLOR)
im2=cv2.imread('heart3.png', cv2.IMREAD_COLOR)
# im2, landmarks2, faces_num = read_im_and_landmarks(im)

while True :
    (grabbed, frame) = camera.read()
    frame = cv2.flip(frame, 1)
    im1, landmarks1, faces_num2 = read_im_and_landmarks(frame)

    # skyImg = cv2.imread('../fig/sky.jpg', -1)
    # rplogoImg = cv2.imread('../fig/raspberry-pi-logo.png', -1)

    ## Image Resize
    rp = cv2.resize(im, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
    rp2 = cv2.resize(im2, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)

    ## Image Addtion with Alpha
    x_offset=landmarks1[2,0]
    y_offset=landmarks1[2,1]

    x_offset2 = landmarks1[14, 0]
    y_offset2 = landmarks1[14, 1]
    # y_offset = 10
    for c in range(0, 3):
        im1[y_offset:y_offset + rp.shape[0], x_offset:x_offset + rp.shape[1], c] = \
            rp[:, :, c] * (rp[:, :, 2] / 255.0) + \
            im1[y_offset:y_offset + rp.shape[0], x_offset:x_offset + rp.shape[1], c] * (1.0 - rp[:, :, 2] / 255.0)

        im1[y_offset2:y_offset2 + rp2.shape[0], x_offset2:x_offset2 + rp2.shape[1], c] = \
            rp2[:, :, c] * (rp2[:, :, 2] / 255.0) + \
            im1[y_offset2:y_offset2 + rp2.shape[0], x_offset2:x_offset2 + rp2.shape[1], c] * (1.0 - rp2[:, :, 2] / 255.0)

        cv2.imshow('Frame', im1 / 255)
    key = cv2.waitKey(1) & 0xFFF
    if key == 27:
        break

camera.release()
        #
    # if faces_num is 0 or faces_num2 is 0:
    #     cv2.imshow('Frame', im1)
    # else:
    #     M = transformation_from_points(landmarks1[ALIGN_POINTS],
    #                                    landmarks2[ALIGN_POINTS])
    #
    #     mask = get_face_mask(im2, landmarks2)
    #     mask2 = get_face_mask(im1, landmarks1)
    #
    #     warped_mask = warp_im(mask, M, im1.shape)
    #     combined_mask = numpy.max([mask2, warped_mask],
    #                               axis=0)
    #
    #     warped_im2 = warp_im(im2, M, im1.shape)
    #     warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)
    #
    #     output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
    #
    #     # type()
    #     # for i in range(68):
    #     #     cv2.circle(output_im,landmarks1[i],1,color="red")
    #
    #     # MM = transformation_from_points(landmarks1[ALIGN_POINTS2],
    #     #                                landmarks2[ALIGN_POINTS2])
    #     #
    #     # mask = get_face_mask(im2, landmarks2)
    #     # mask2 = get_face_mask(im1, landmarks1)
    #     #
    #     # warped_mask = warp_im(mask, MM, im1.shape)
    #     # combined_mask = numpy.max([mask2, warped_mask],
    #     #                           axis=0)
    #     #
    #     # warped_im2 = warp_im(im2, MM, im1.shape)
    #     # warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)
    #     # output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
    #     cv2.imshow('Frame', output_im / 255)
    #
    # key = cv2.waitKey(1) & 0xFFF
    #
    # if key == 27:
    #     break

# camera.release()
