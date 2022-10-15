# define class and related functions
from copy import copy, deepcopy
from dataclasses import dataclass, field
import numpy as np
import PyQt5.QtCore
from PyQt5.QtWidgets import QMessageBox
from abc import ABC, abstractmethod
from modules import interface
from modules.utility import *

class ImageOperation(ABC):
    def __init__(self, image):
        self.image = image
        self.image_backup = deepcopy(image)

    def undo(self):
        self.image = self.image_backup

    @abstractmethod
    def execute(self):
        pass


# frozen = True means that the class cannot be modified
# kw_only = True means that the class cannot be instantiated with positional arguments
@dataclass(frozen = True)
class Image:

    data: np.ndarray  # required on init
    path: str = ''
    metadata: dict = field(default_factory=dict)
    operations_dict = {}

    # def append_operation(self, operation):
    #     self.operations_dict[operation.__class__.__name__] = operation
    
    def clear_operations(self):
        self.operations_dict = {}
    
    def run_processing(self):
        for operation in self.operations_dict.values():
            operation.execute()

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

