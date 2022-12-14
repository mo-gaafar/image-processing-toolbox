"""Interpolation functions for image scaling and rotation opeartions
Created on 2022/11
Author: M. Nasser Gaafar

Functions:
    linear_interp(p1, p2, px): linear interpolation of two points
    bilinear_interp_pixel(x, y, img): bilinear interpolation of a pixel
    special_round(x): rounds x to the nearest integer, but if x is between two integers, it rounds to the nearest even integer
    nn_interp_pixel(x, y, img): nearest neighbor interpolation of a pixel

"""

import numpy as np
from modules.image import *


def linear_interp(p1, p2, px):
    return p1 * (1 - px) + p2 * px


def bilinear_interp_pixel(x, y, img):
    '''Bilinear interpolation of a pixel'''
    '''x and y are the relative coordinates of the pixel in original image'''

    x1 = int(np.floor(x))
    x2 = int(np.ceil(x))
    y1 = int(np.floor(y))
    y2 = int(np.ceil(y))

    # p1 -- p' ---- p2
    # |     |       |
    # |     |       |
    # |     |       |
    # p3 --p''---- p4

    # check if p1,p2,p3,p4 are out of bounds
    if x1 < 0 or x2 >= np.shape(img)[0] or y1 < 0 or y2 >= np.shape(img)[1]:
        return 0
        # if x2 >= np.shape(img)[0]:
        #     x2 = x1
        # if y2 >= np.shape(img)[1]:
        #     y2 = y1

    # x is rows, y is columns in the image
    # axis 0 -> rows

    p1 = img[x1, y1]
    p2 = img[x1, y2]
    p3 = img[x2, y1]
    p4 = img[x2, y2]

    # calculate the new pixel value
    return linear_interp(linear_interp(p1, p2, y - y1),
                         linear_interp(p3, p4, y - y1), x - x1)


def special_round(x):
    '''Special rounding function for nearest neighbor
    rounds down the value if the fractional part is 0.5 or less'''
    if x - int(x) <= 0.5:
        return int(x)
    else:
        return int(x) + 1


def nn_interp_pixel(x, y, img):
    '''Nearest neighbor interpolation of a pixel'''
    '''x and y are the relative coordinates of the pixel in original image'''

    x1 = special_round(x)
    y1 = special_round(y)

    # check if p1 is out of bounds
    if x1 < 0 or y1 < 0 or x1 >= np.shape(img)[0] or y1 >= np.shape(img)[1]:
        return 0

    return img[x1, y1]
