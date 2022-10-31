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


def reset_image(self):
    '''Resets the image to its original size'''
    try:
        # undo previous operations
        self.image1.clear_operations()
        selected_window = int(self.interpolate_output_combobox.currentIndex())
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

    def clear_operations(self, clear_backup=False):
        if clear_backup == True and self.image_backup is not None:
            self.image_backup = None
        self.operations_dict = {}
        self.undo()

    def clear_image_data(self):
        self.data = None
        self.path = ''
        self.metadata = {}
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

    def get_formatted_metadata(self):
        f_metadata = ''
        for key, value in self.metadata.items():
            f_metadata += key + ': ' + str(value) + '\n'
        print(f_metadata)
        return f_metadata

    def get_processing_time(self):
        return self.last_processing_time_ms

    def get_img_format(self):
        return self.path.split('.')[-1]
