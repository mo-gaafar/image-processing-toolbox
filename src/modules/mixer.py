# define class and related functions
from copy import copy, deepcopy
from dataclasses import dataclass, field
from msilib.schema import Component
from typing import ClassVar
from scipy.fft import fftfreq, rfftn, irfftn, fft2, ifft2, fftn, ifftn, fftshift
import numpy as np
from sympy import fourier_transform
from modules.utility import *
from modules import interface
import PyQt5.QtCore
from PyQt5.QtWidgets import QMessageBox

# frozen = True means that the class cannot be modified
# kw_only = True means that the class cannot be instantiated with positional arguments


@dataclass
class ImageFFT():

    image_data: np.array

    fftfreq: np.ndarray = None
    fftphase: np.ndarray = None
    fftmag: np.ndarray = None
    uniform_phase: np.ndarray = None
    uniform_magnitude: np.ndarray = None
    fftreal: np.ndarray = None
    fftimag: np.ndarray = None

    fftdata: np.array = None

    def __post_init__(self):
        self.process_image()

    def process_image(self):
        self.fftdata = rfftn(self.image_data)

        # self.fftfreq = fftfreq(self.data.size, 1 / 44100)
        self.fftreal = np.real(self.fftdata)
        self.fftimag = np.imag(self.fftdata)
        self.fftmag = np.abs(self.fftdata)
        self.fftphase = np.exp(np.multiply(1j, np.angle(self.fftdata)))

        # TODO: check if this is correct
        self.uniform_phase = self.fftmag * np.exp(0)
        self.uniform_magnitude = 100 * self.fftphase


@ dataclass
class Image():

    data: np.ndarray  # required on init
    path: str = ''
    image_height: int = 0
    image_width: int = 0
    image_depth: int = 0
    fourier_enable: bool = False
    image_fft: ImageFFT = None

    def __post_init__(self):
        self.update_parameters()
        if (self.fourier_enable):
            self.init_fourier()

    def init_fourier(self):
        self.image_fft = ImageFFT(image_data=self.data)

    def update_parameters(self):
        # TODO: calculate basic parameters (width, height, etc)
        self.image_height = self.data.shape[0]
        self.image_width = self.data.shape[1]
        self.image_depth = self.data.shape[2]

    def get_image_data(self):
        return self.data


# contains all the fourier transformed data of an image
# on performs fft2 on initialization
@ dataclass
class ImageConfiguration():
    image: Image
    selected_feature: str = "Full"
    selected_feature_index: int = 6
    strength_percent: int = 100

    selected_feature_dict: ClassVar[dict] = {
        "Phase": 0,
        "Magnitude": 1,
        "Uniform Phase": 2,
        "Uniform Magnitude": 3,
        "Real": 4,
        "Imaginary": 5,
        "Full": 6,
        "FT Magnitude": 7,
        "FT Phase": 8,
        "FT Real": 9,
        "FT Imaginary": 10
    }
    ''' Global to all class instances'''

    def set_weight(self, weight):
        self.strength_percent = weight

    def set_selected_feature(self, feature=None, index=None):
        '''Sets the selected feature based on its index or name'''
        if feature is not None:
            self.selected_feature_index = self.convert_feature_to_index(
                feature)
            self.selected_feature = feature

        elif index is not None:
            self.selected_feature = self.convert_index_to_feature(index)
            self.selected_feature_index = index
        else:
            raise Exception("Invalid Inputs")

    def convert_feature_to_index(self, feature):
        if feature in self.selected_feature_dict:
            return self.selected_feature_dict[feature]
        else:
            raise Exception("Invalid Feature")

    def convert_index_to_feature(self, index):
        for key, value in self.selected_feature_dict.items():
            if value == index:
                return key
        raise Exception("Invalid Index")

    def get_original_image(self):
        return self.image.data

    # my sincerest apologies for this function
    def get_selected_in_ft(self):
        if(self.selected_feature == "Phase"):
            image = copy(self.image.image_fft.fftphase)
        elif(self.selected_feature == "Magnitude"):
            image = copy(self.image.image_fft.fftmag)
        elif(self.selected_feature in ["Uniform Phase", "FT Magnitude"]):
            image = copy(self.image.image_fft.uniform_phase)
        elif(self.selected_feature in ["Uniform Magnitude", "FT Phase"]):
            image = copy(self.image.image_fft.uniform_magnitude)
        elif(self.selected_feature in ["Real", "FT Real"]):
            image = copy(self.image.image_fft.fftreal)
        elif(self.selected_feature in ["Imaginary", "FT Imaginary"]):
            image = copy(self.image.image_fft.fftimag)
        elif(self.selected_feature == "Full"):
            image = copy(self.image.image_fft.fftdata)
        else:
            raise Exception("Invalid Feature")

        return image

    def get_weighted_in_ft(self):
        # TODO add phase weighting in uniform magnitude ??
        # Phase special case weighting
        if self.selected_feature == "Phase":
            weighted_phase = np.exp(1j * np.angle(
                self.image.image_fft.fftdata) * self.strength_percent/100)
            return weighted_phase
        else:
            return self.get_selected_in_ft() * self.strength_percent / 100

    def get_ft_plot(self):
        if self.selected_feature in ["FT Phase"]:
            return np.real(
                fftshift(self.get_selected_in_ft()))
        elif self.selected_feature in ["FT Magnitude", "FT Real", "FT Imaginary"]:
            return np.multiply(np.log10(np.abs(
                fftshift(self.get_selected_in_ft()))), 20)
        else:
            print_debug("Invalid FT Feature")
            return self.get_selected_in_ft()

    def get_processed_image(self):
        '''Selects image based on required feature then \n
        internaly weighs images based on the strength_percent'''

        image = self.get_selected_in_ft()
        # modify magnitude component and restore phase
        # weighted_image_fft = np.multiply(np.multiply(np.abs(
        #     image), self.strength_percent / 100), np.exp(np.multiply(np.angle(image), 1j)))

        # TODO: convert from FFT coefficients to image after applying weights
        weighted_image = irfftn(self.get_weighted_in_ft())

        return weighted_image


class ImageMixer():
    def __init__(self, selection1: ImageConfiguration = None,
                 selection2: ImageConfiguration = None) -> None:
        self.selected_images = [deepcopy(selection1), deepcopy(selection2)]
        self.mixed_image: Image = None

    def set_selection_feature(self, selection_index: int, feature_idx: int = None, feature_name: str = None):
        if feature_idx == None:
            self.selected_images[selection_index].set_selected_feature(
                feature=feature_name)
        else:
            self.selected_images[selection_index].set_selected_feature(
                index=feature_idx)

    def set_selection_weight(self, selection_index: int, weight: int):
        self.selected_images[selection_index].set_weight(weight)

    def mix_images(self):
        '''Mix images based on selected features in the frequency domain'''

        # if all is good then mix
        temp1 = self.selected_images[0].get_weighted_in_ft()
        temp2 = self.selected_images[1].get_weighted_in_ft()

        # if temp1.selected_feature == "Phase" and temp2.selected_feature == "Phase":
        # mixed_image = np.multiply(temp1, temp2)
        # additive weighted mixing for imag , real

        if (self.selected_images[0].selected_feature and
                self.selected_images[1].selected_feature in ["Phase", "Magnitude", "Uniform Phase", "Uniform Magnitude"]):
            mixed_image = np.multiply(temp1, temp2)
        elif (self.selected_images[0].selected_feature and self.selected_images[1].selected_feature in ["Imaginary", "Real", "Full"]):
            if self.selected_images[0].selected_feature == "Imaginary":
                temp1 = np.multiply(temp1, 1j)
            elif self.selected_images[1].selected_feature == "Imaginary":
                temp2 = np.multiply(temp2, 1j)

            mixed_image = np.add(temp1, temp2)
        else:
            # TODO this should not exist
            mixed_image = np.add(temp1, temp2)
            print("Invalid Feature Combination in Mixer")

        mixed_image_data = np.abs(
            irfftn(mixed_image))

        self.mixed_image = Image(data=mixed_image_data)

    def get_mixed_image_data(self):
        return self.mixed_image.data

# TODO increase class responsibilities


def update_mixer(self):
    '''Component 1'''
    # selected image from dropdown
    selection1_idx = self.mixer_component1_comboBox.currentIndex()
    # selected feature from dropdown
    selection1_feature_idx = self.mixed_component1_comboBox.currentText()
    # strength slider
    selection1_weight = self.mixer_component1_horizontalSlider.value()

    '''Component 2'''
    # selected image from dropdown
    selection2_idx = self.mixer_component2_comboBox.currentIndex()
    # selected feature from dropdown
    selection2_feature_idx = self.mixed_component2_comboBox.currentText()
    # strength slider
    selection2_weight = self.mixer_component2_horizontalSlider.value()

    # check if selected images exist
    if ((selection1_idx == 0) or (selection2_idx == 0)) and (self.image1_configured == None):
        print_debug("Image 1 does not exist, aborting...")
        return
    if ((selection1_idx == 1) or (selection2_idx == 1)) and (self.image2_configured == None):
        print_debug("Image 2 does not exist, aborting...")
        return

    # set selected image local references
    if selection1_idx == 0:
        selection1 = self.image1_configured
    elif selection1_idx == 1:
        selection1 = self.image2_configured
    if selection2_idx == 0:
        selection2 = self.image1_configured
    elif selection2_idx == 1:
        selection2 = self.image2_configured

    # check if both selections are equal in size
    if selection1.image.data.shape != selection2.image.data.shape:
        print_debug("Images are not of same size, aborting...")
        # raise Exception("Images are not of equal size")
        return

    # reinitialize mixer using copies of selected images (done internally in init)
    self.mixer = ImageMixer(selection1, selection2)
    self.mixer.set_selection_feature(0, feature_name=selection1_feature_idx)
    self.mixer.set_selection_feature(1, feature_name=selection2_feature_idx)

    # set weights
    self.mixer.set_selection_weight(0, selection1_weight)
    self.mixer.set_selection_weight(1, selection2_weight)

    # mix images
    self.mixer.mix_images()

    # selected output label dropdown
    output_idx = self.mixer_output_comboBox.currentIndex()

    # update selected output
    if output_idx == 0:
        interface.update_display(self, display_keys=["output1"])
    elif output_idx == 1:
        interface.update_display(self, display_keys=["output2"])

    pass
