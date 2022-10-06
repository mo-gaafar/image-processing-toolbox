# define class and related functions
from copy import copy, deepcopy
from dataclasses import dataclass, field
from msilib.schema import Component
from typing import ClassVar
import numpy as np
from modules.utility import *
from modules import interface
import PyQt5.QtCore
from PyQt5.QtWidgets import QMessageBox

# frozen = True means that the class cannot be modified
# kw_only = True means that the class cannot be instantiated with positional arguments



#TODO: add a function to update the image in the viewer
#TODO: add exception handling e.g. empty image
#TODO: add exception handling e.g. wrong image type + logging

#TODO: plan how to handle different images in the architecture
#TODO: find a dicom viewer
#TODO: supported types of images (bmp, jpg, dicom)
@ dataclass
class Image():

    data: np.ndarray  # required on init
    path: str = ''
    image_height: int = 0
    image_width: int = 0
    image_depth: int = 0

    def __post_init__(self):
        self.update_parameters()

    def update_parameters(self):
        # TODO: calculate basic parameters (width, height, etc)
        self.image_height = self.data.shape[0]
        self.image_width = self.data.shape[1]
        self.image_depth = self.data.shape[2]
    # TODO: 
    def get_image_data(self):
        return self.data

class DICOMImage(Image):
    def __post_init__(self):
        self.update_parameters()

    def update_parameters(self):
        pass

@ dataclass
class ImageMetadata():

    path: str = ''
    image_height: int = 0
    image_width: int = 0
    image_depth: int = 0

