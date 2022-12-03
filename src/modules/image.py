# define class and related functions
from copy import copy, deepcopy
from dataclasses import dataclass, field
import numpy as np
import PyQt5.QtCore
from PyQt5.QtWidgets import QMessageBox
import time
from abc import ABC, abstractmethod
from modules import interface
from modules import image
from modules.utility import *


def restore_original(self):
    '''Gets the original image'''
    try:
        # undo previous operations
        self.image1 = deepcopy(self.safe_image_backup)
        selected_window = interface.get_user_input(self)['output window']
        interface.display_pixmap(self,
                                 image=self.image1,
                                 window_index=selected_window)
        interface.update_img_resize_dimensions(self, 'resized',
                                               self.image1.get_pixels())
        interface.print_statusbar(self, 'Restores Original Image')

    except:
        QMessageBox.critical(self, 'Error',
                             'Error Running Operation: No Image Loaded')
        return
    # refresh the display


def reset_image(self):
    '''Resets the image to its original state'''
    try:
        # undo previous operations
        self.image1.clear_operations()
        selected_window = interface.get_user_input(self)['output window']
        interface.display_pixmap(self,
                                 image=self.image1,
                                 window_index=selected_window)
        interface.update_img_resize_dimensions(self, 'resized',
                                               self.image1.get_pixels())
        interface.print_statusbar(self, 'Image Reset')

    except:
        QMessageBox.critical(self, 'Error',
                             'Error Running Operation: No Image Loaded')
        return
    # refresh the display


class ImageOperation(ABC):
    '''Abstract class for image operations'''

    def __init__(self):
        self.image = None
        self.image_backup = None

    def configure(self, factor):
        self.factor = factor
        return self

    def set_image(self, image):
        self.image = image
        self.image_backup = deepcopy(image)

    def undo(self):
        self.image = self.image_backup

    @abstractmethod
    def execute(self, image):
        pass


# frozen = True means that the class cannot be modified
# kw_only = True means that the class cannot be instantiated with positional arguments


@dataclass(frozen=False)
class Image:

    data: np.ndarray  # required on init
    path: str = ''
    metadata: dict = field(default_factory=dict)
    operations_dict = {}
    image_backup = None
    last_processing_time_ms = 0
    image_fft = None
    alloc_dtype = None

    def clear_operations(self, undo_old=True, clear_backup=False):
        ''' Clears all operations from the image.

        undo_old: if True, the image is restored to its original state
        clear_backup: if True, the backup image is cleared

        '''
        if clear_backup == True and self.image_backup is not None:
            self.image_backup = None
        self.operations_dict = {}
        if undo_old == True:
            self.undo()

    def clear_image_data(self):
        self.data = None
        self.path = ''
        self.metadata = {}
        self.alloc_dtype = None
        self.operations_dict = {}
        self.image_backup = None
        self.last_processing_time_ms = 0

    def add_operation(self, operation):
        self.operations_dict[operation.__class__.__name__] = operation

    def run_processing(self):
        tic = time.perf_counter()

        # processing pipeline
        self.image_backup = deepcopy(self)
        self.image_out = None
        for operation in self.operations_dict.values():
            operation.set_image(self)
            self.image_out = operation.execute()
            self.data = self.image_out.data
            self.metadata = self.image_out.metadata
            print_log('Processing image with ' + operation.__class__.__name__)
        toc = time.perf_counter()
        self.last_processing_time_ms = int((toc - tic) * 1000)

    def undo(self):
        if self.image_backup is not None:
            self.data = self.image_backup.data

            # self.operations_dict = self.image_backup.operations_dict

    def get_pixels(self):
        print_debug("Getting image pixels")
        print_debug(self.data)
        return self.data

    def get_metadata(self):
        return self.metadata

    def get_histogram(self, relative=True):
        """
        Returns the histogram of the image.

        Args:
            relative (bool): if True, the histogram is normalized to sum to 1

        Returns:
            histogram (np.ndarray): the histogram of the image
            range (tuple): the range of the histogram

        """
        # get range from allocated depth
        L = 2**self.get_channel_depth()
        range_histo = (0, L - 1)
        # create empty array for the histogram
        histogram = np.zeros(range_histo[1] - range_histo[0] + 1)
        sum = 0

        height, width = np.shape(self.data)

        for x in range(height):
            for y in range(width):
                histogram[self.data[x, y]] += 1
                sum += 1

        if relative == True:
            histogram = histogram / sum

        return histogram, range_histo

    def get_formatted_metadata(self):
        f_metadata = ''
        for key, value in self.metadata.items():
            f_metadata += key + ': ' + str(value) + '\n'
        print(f_metadata)
        return f_metadata

    def get_processing_time(self):
        return self.last_processing_time_ms

    def get_channel_depth(self):
        # calculate the number of bits per pixel using log2 and max
        depth = int(np.log2(np.max(self.data)) + 1)
        print_log('Calculating channel depth' + str(depth))
        return depth

    def get_alloc_pixel_dtype(self):
        # saved only once
        # if self.alloc_dtype is None:
        #     self.alloc_dtype = self.data.dtype
        #     return self.data.dtype
        # else:
        #     return self.alloc_dtype
        return np.uint32

    def get_fft(self):
        """Returns an fft object of the image. 
        If the image has not been transformed yet, it is transformed and saved in the object.
        If the image has been fft transformed, the saved object is returned.
        However, if the image has been transformed and then modified, the fft is recalculated.
        """
        if self.image_fft is None:
            self.image_fft = ImageFFT(image_ref=self)
        else:
            if self.image_fft.image_ref is not self:
                self.image_fft = ImageFFT(image_ref=self)

        return self.image_fft

    def get_img_format(self):
        return self.path.split('.')[-1]


@dataclass(frozen=True)
class ImageFFT:
    """Class for storing the FFT of an image"""
    real: np.ndarray
    imaginary: np.ndarray
    magnitude: np.ndarray
    phase: np.ndarray
    image_ref: Image = None
    fft_data: np.ndarray = field(default_factory=np.ndarray)

    def __post_init__(self):
        self.fft_data = np.fft.fft2(self.image_data)
        self.real = self.fft_data.real
        self.imaginary = self.fft_data.imag
        self.magnitude = np.abs(self.fft_data)
        self.phase = np.angle(self.fft_data)

    def get_real(self):
        return self.real

    def get_imaginary(self):
        return self.imaginary

    def get_magnitude(self):
        return self.magnitude

    def get_phase(self):
        return self.phase

    def get_fft(self):
        return self.real + 1j * self.imaginary

    def get_fft_conj(self):
        return self.real - 1j * self.imaginary

    def get_fft_magnitude(self):
        return self.magnitude

    def get_fft_phase(self):
        return self.phase

    def get_fft_magnitude_conj(self):
        return self.magnitude

    def get_fft_phase_conj(self):
        return -self.phase
