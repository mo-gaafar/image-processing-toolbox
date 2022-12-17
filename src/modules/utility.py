""" Contains utility functions for image processing.
Created on 2022/11
Author: M. Nasser Gaafar

Functions:
    print_debug(string): prints and logs based on preset variables in util
    print_log(string): logs based on preset variables in util
    map_range(value, in_min, in_max, out_min, out_max): maps a value from one range to another
    uniform_padding(array, n, val): pads array with val, n times on 4 sides
    merge_sort(in_array): merge sort algorithm

Globals:
    DEBUG_MODE: if True, prints debug messages in terminal and also logs them
    LOGGING_MODE: if True, logs messages in file only
    
"""

import logging
import numpy as np

# utility globals
DEBUG_MODE = False
LOGGING_MODE = True


logging.basicConfig(filename="logs.log",
                    format='%(asctime)s %(message)s', filemode='w')

logger = logging.getLogger()

# set the threshold to debug
logger.setLevel(logging.INFO)

logger.debug("Logger Initialized")


def print_debug(string):
    '''This prints and logs based on preset variables in util'''
    if DEBUG_MODE:
        print(string)
    if LOGGING_MODE:
        logger.info(string)


def print_log(string):
    if LOGGING_MODE:
        logger.info(string)


def map_range(value, in_min, in_max, out_min, out_max):
    return out_min + (((value - in_min) / (in_max - in_min)) * (out_max - out_min))


def uniform_padding(array, n, val):
    """
    Pads array with val, n times on 4 sides.

    array: input array
    n: size of padding in all sides (per side)
    val: value to pad with
    """
    n = int(n)
    out_arr = np.zeros((array.shape[0] + 2*n, array.shape[1] + 2*n))

    # loop on new array and fill with image and padding
    for x in range(0, out_arr.shape[0]):
        for y in range(0, out_arr.shape[1]):
            if x < n or x >= out_arr.shape[0] - n or y < n or y >= out_arr.shape[1] - n:
                out_arr[x, y] = val
            else:
                out_arr[x, y] = array[x-n][y-n]

    return out_arr


def merge_sort(in_array):
    """ 
    Merge sort algorithm.

    in_array: input array

    Returns sorted array
    """
    array = in_array
    if len(array) > 1:
        mid = len(array) // 2
        left = array[:mid]
        right = array[mid:]

        merge_sort(left)
        merge_sort(right)

        i = j = k = 0

        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                array[k] = left[i]
                i += 1
            else:
                array[k] = right[j]
                j += 1
            k += 1

        while i < len(left):
            array[k] = left[i]
            i += 1
            k += 1

        while j < len(right):
            array[k] = right[j]
            j += 1
            k += 1
    return array


def get_median(array):
    """Gets median of array using merge sort.

    array: input array

    returns median of array
    """

    # if array is 2d then flatten it
    flat_array = array
    if len(np.size(array)) == 2:
        flat_array = array.flatten()

    sorted_arr = merge_sort(flat_array)

    mid = 0  # median index
    if len(sorted_arr) % 2 == 0:
        # if even then get avg of 2 middle elements
        mid = len(sorted_arr) // 2
        return (sorted_arr[mid] + sorted_arr[mid-1]) / 2
    else:
        # if odd then get middle element
        mid = round(len(sorted_arr) / 2) - 1
        return sorted_arr[mid]


def round_nearest_odd(value):
    """Rounds value to nearest odd number.

    value: input value

    returns rounded value
    """
    return int(np.ceil(value) // 2 * 2 + 1)


def complex_abs(array):
    """Gets absolute value of complex array.

    array: input array

    returns absolute value of array
    """
    return np.sqrt(np.power(np.real(array), 2) + np.power(np.imag(array), 2))


def complex_angle(array):
    """Gets angle of complex array.

    array: input array

    returns angle of array
    """
    return np.arctan2(np.imag(array), np.real(array))


def get_histogram(image_arr):
    """Gets histogram of image.

    Params:
        image_arr: image array
    Returns:
        histogram: image histogram
    """
    # get image levels using log2
    L = int(2 ** np.ceil(np.log2(np.max(image_arr))))

    # get image histogram
    histogram = np.zeros(L)
    for i in range(image_arr.shape[0]):
        for j in range(image_arr.shape[1]):
            histogram[image_arr[i][j]] += 1

    return histogram


def histo_mean(image_arr=None, histogram=None):
    """Gets mean of image histogram.

    Params:
        image_arr: image array
        OR 
        histogram: image histogram
    Returns:
        mean: mean of image histogram
    """

    # get image histogram
    if histogram is None and image_arr is not None:
        histogram = get_histogram(image_arr)

    if histogram is None:
        raise Exception("No histogram or image array provided")

    # get mean of histogram
    mean = 0
    for i in range(len(histogram)):
        mean += i * histogram[i]
    mean = mean / np.sum(histogram)

    return mean


def histo_std_dev(image_arr=None, histogram=None):
    """Gets standard deviation of image histogram.

    Params:
        image_arr: image array
        OR 
        histogram: image histogram
    Returns:
        std_dev: standard deviation of image histogram
    """

    # get image histogram
    if histogram is None and image_arr is not None:
        histogram = get_histogram(image_arr)

    if histogram is None:
        raise Exception("No histogram or image array provided")

    # get mean of histogram
    mean = histo_mean(histogram=histogram)

    # get standard deviation
    std_dev = 0
    for i in range(len(histogram)):
        std_dev += (i - mean)**2 * histogram[i]
    std_dev = std_dev / np.sum(histogram)
    std_dev = np.sqrt(std_dev)

    return std_dev


def clip(arr, min_val, max_val):
    """ Clips array values between min_val and max_val.

    Params:
        arr: input array
        min_val: minimum value
        max_val: maximum value
    Returns:
        clipped array

    """

    # loop on array and clip values
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i][j] < min_val:
                arr[i][j] = min_val
            elif arr[i][j] > max_val:
                arr[i][j] = max_val

    return arr


# def rescale_intensity(arr, depth = 1, shift_min = False):
#     """ Intensity scaling.
#     Params:
#         depth: channel depth
#         shift_min: whether to shift by minimum value or truncate negatives
#     """

#     if shift_min:
#         pass
