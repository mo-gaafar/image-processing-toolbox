import numpy as np
from modules.image import *
from modules import interface


def resize_image(self):
    '''Resizes the image to the specified dimensions'''
    # # get the new dimensions
    # width = self.resize_width_spinBox.value()
    # height = self.resize_height_spinBox.value()

    # get user input parameters data
    # TODO: create a getter function that returns a params dict from the UI
    factor = self.resize_spinbox.value()

    # get the selected interpolator
    interpolator = read_interpolator(self.interpolate_combobox.currentText())
    if interpolator == None:
        return
    # configure the resize operation
    resize_operation = interpolator.__post_init__(self.image1, width, height)
    # add the operation to the image
    self.image1.append_operation(resize_operation)
    # run the processing 
    self.image1.run_processing()
    # refresh the display
    interface.refresh_display(self)


def read_interpolator(interpolator_name) -> ImageOperation:
    # array of supported interpolators
    interpolators = {
        'Nearest Neighbor': NearestNeighborInterpolator(),
        'Bilinear': BilinearInterpolator(),
        'None': None
    }
    if interpolator_name in interpolators:
        return interpolators[interpolator_name]
    else:
        raise Warning("Unsupported interpolator")


class BilinearInterpolator(ImageOperation):
    def __init__(self, image):
        super().__init__(image)

    def interpolate(self, x, y):
        pass

    def execute(self):
        self.image.data = self.interpolate(self.image.data)


class NearestNeighborInterpolator(ImageOperation):
    def __init__(self, image):
        super().__init__(image)

    def interpolate(self, x, y):
        pass

    def execute(self):
        self.image.data = self.interpolate(self.image.data)
