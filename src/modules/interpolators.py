import numpy as np
from modules.image import *
from modules import interface

# connected to the apply button in resize tab


def resize_image(self):
    '''Resizes the image to the specified dimensions'''

    # get user input parameters data
    factor = interface.get_user_input()['resize factor']

    # get the selected interpolator class
    interpolator = read_interpolator(
        interface.get_user_input()['interpolation method'])

    if interpolator == None:
        return

    # configure the resize operation object
    resize_operation = interpolator.configure(factor)

    # add the operation to the image
    self.image1.add_operation(MonochoromeConversion())
    self.image1.add_operation(resize_operation)
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

    def linear_interp(self, x, x1, x2, y1, y2):
        return y1 + (x - x1) * (y2 - y1) / (x2 - x1)

    def interpolate(self, image):
        # get the image dimensions
        height, width = image.shape
        # create a new image with the new dimensions
        new_image = np.zeros((self.factor * height, self.factor * width))
        # loop through the new image and interpolate the values
        for i in range(0, new_image.shape[0]):
            for j in range(0, new_image.shape[1]):
                # get the new image coordinates
                x = i / self.factor
                y = j / self.factor
                # get the coordinates of the nearest neighbors
                x1 = int(np.floor(x))
                x2 = int(np.ceil(x))
                y1 = int(np.floor(y))
                y2 = int(np.ceil(y))
                # get the pixel values of the nearest neighbors
                q11 = image[x1, y1]
                q12 = image[x1, y2]
                q21 = image[x2, y1]
                q22 = image[x2, y2]
                # interpolate the pixel value
                new_image[i, j] = self.linear_interp(x, x1, x2, self.linear_interp(
                    y, y1, y2, q11, q12), self.linear_interp(y, y1, y2, q21, q22))
        return new_image

    def execute(self):
        self.image.data = self.interpolate(self.image.data)


class NearestNeighborInterpolator(ImageOperation):
    def __init__(self, image):
        super().__init__(image)

    def interpolate(self, image):
        # get the image dimensions
        height, width = image.shape
        # create a new image with the new dimensions
        new_image = np.zeros((self.factor * height, self.factor * width))
        # loop through the new image and interpolate the values
        for i in range(0, new_image.shape[0]):
            for j in range(0, new_image.shape[1]):
                # get the new image coordinates
                x = i / self.factor
                y = j / self.factor
                # get the coordinates of the nearest neighbors
                x1 = int(np.floor(x))
                y1 = int(np.floor(y))
                # get the pixel values of the nearest neighbors
                q11 = image[x1, y1]
                # interpolate the pixel value
                new_image[i, j] = q11
        return new_image

    def execute(self):
        self.image.data = self.interpolate(self.image.data)
