# define class and related functions
from copy import copy, deepcopy
from dataclasses import dataclass, field
import numpy as np
import PyQt5.QtCore
from PyQt5.QtWidgets import QMessageBox
from abc import ABC, abstractmethod
from modules import interface
from modules.image import Image
from modules.utility import *


class ImageOperation(ABC):
    '''Abstract class for image operations'''

    def __init__(self):
        self.image = None
        self.image_backup = None

    def configure(self, factor):
        self.factor = factor
        return self

    def undo(self):
        self.image = self.image_backup

    @abstractmethod
    def execute(self, image) -> Image:
        pass


class MonochoromeConversion(ImageOperation):

    def execute(self):
        # calculate mean over image channels (color depth axis = 2)
        self.image.data = self.image.data.mean(axis=2)
        # quantizing into 256 levels
        self.image.data = self.image.data.astype(np.uint8)


# frozen = True means that the class cannot be modified
# kw_only = True means that the class cannot be instantiated with positional arguments
@dataclass(frozen=False)
class Image:

    data: np.ndarray  # required on init
    path: str = ''
    metadata: dict = field(default_factory=dict)
    operations_dict = {}

    # def append_operation(self, operation):
    #     self.operations_dict[operation.__class__.__name__] = operation

    def clear_operations(self):
        self.operations_dict = {}

    def add_operation(self, operation):
        self.operations_dict[operation.__class__.__name__].append(operation)

    def run_processing(self):
        # processing pipeline
        self.image_backup = deepcopy(self)
        for operation in self.operations_dict:
            operation.execute(self)

    def get_pixels(self):
        print_debug(self.data)
        return self.data

    def get_metadata(self):
        return self.metadata

    def get_formatted_metadata(self):
        f_metadata = ''
        for key, value in self.metadata.items():
            f_metadata += key + ': ' + str(value) + '\n'
        print(f_metadata)
        return f_metadata

    def get_img_format(self):
        return self.path.split('.')[-1]
