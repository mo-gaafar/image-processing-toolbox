'''Should contain printdebug and logging functionality '''
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


def mapRange(value, inMin, inMax, outMin, outMax):
    return outMin + (((value - inMin) / (inMax - inMin)) * (outMax - outMin))


def uniform_padding(array, n, val):
    """
    Pads array with val, n times on 4 sides.

    array: input array
    n: size of padding in all sides (per side)
    val: value to pad with
    """

    out_arr = []
    for i in range(0, array.shape[0]+2*n):
        for j in range(0, array.shape[1]+2*n):
            out_arr[i][j] = val

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            out_arr[i+n][j+n] = array[i][j]

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
