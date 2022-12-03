from modules.image import Image, ImageOperation
from dataclasses import dataclass, field
import numpy as np
from copy import copy, deepcopy
from abc import ABC, abstractmethod
from modules import interface
from modules import image
from modules.utility import *


@dataclass(frozen=False)
class ImageFFT:
    """Class for storing the FFT of an image"""
    real: np.ndarray = None
    imaginary: np.ndarray = None
    magnitude: np.ndarray = None
    phase: np.ndarray = None
    fft_data: np.ndarray = None
    image_ref: Image = field(default_factory=np.ndarray)
    image_curr_data = None

    def __post_init__(self):
        self.fft_data = np.fft.fft2(self.image_ref.data)
        self.image_curr_data = self.image_ref.data
        self.real = self.fft_data.real
        self.imaginary = self.fft_data.imag
        self.magnitude = complex_abs(self.fft_data)
        self.phase = complex_angle(self.fft_data)

    def get_real(self):
        return self.real

    def get_imaginary(self):
        return self.imaginary

    def get_magnitude(self):
        return self.magnitude

    def get_phase(self):
        return self.phase

    def get_fft(self):
        """gets or updates the fft of the image"""
        if self.image_curr_data is not self.image_ref.data or self.fft_data is None:
            self.__post_init__()  # calls __post_init__ again to update the fft data

        return self.fft_data

    def process_fft_displays(self):
        """
        Processes the fft data for display.

        Returns:
            tuple of np.ndarray: The processed fft data for display.
                [0] -> fft_mag_displays: (shifted_logged, shifted, original)
                [1] -> fft_phase_displays: (shifted_logged, shifted, original)
        """
        # get the fft data
        fft_data = self.get_fft()

        # get the magnitude plots
        fft_data_mag = np.abs(fft_data)
        fft_mag_shifted = np.fft.fftshift(fft_data_mag)
        fft_mag_shifted_logged = np.log(fft_mag_shifted + 1)

        # get the phase plots
        fft_data_phase = np.angle(fft_data)
        fft_phase_shifted = np.fft.fftshift(fft_data_phase)
        fft_phase_shifted_logged = np.log(fft_phase_shifted + 2*np.pi)

        fft_mag_displays = (fft_mag_shifted_logged,
                            fft_mag_shifted, fft_data_mag)

        fft_phase_displays = (fft_phase_shifted_logged,
                              fft_phase_shifted, fft_data_phase)

        return fft_mag_displays, fft_phase_displays


class UpdateFFT(ImageOperation):
    """Updates the fft data"""

    def execute(self):
        '''Updates the fft data.

        If the image has not been transformed yet, it is transformed and saved in the object.
        If the image has been fft transformed, the saved object is returned.
        However, if the image has been transformed and then modified, the fft is recalculated.'''
        if self.image.image_fft is None:
            self.image.image_fft = ImageFFT(image_ref=self.image)
        else:
            if self.image.image_fft.image_ref is not self:
                self.image.image_fft = ImageFFT(image_ref=self.image)
            else:
                self.image.image_fft.get_fft()

        return self.image

    def __str__(self):
        return "Update FFT"
