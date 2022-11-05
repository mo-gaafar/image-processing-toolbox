import numpy as np
import threading
from modules.image import *
from modules import interface
from modules.interpolators import *


class MonochoromeConversion(ImageOperation):

    def execute(self):
        # calculate mean over image channels (color depth axis = 2)
        if self.image.data.ndim == 3:
            self.image.data = np.mean(self.image.data, axis=2)
        # quantizing into 256 levels
        self.image.data = self.image.data.astype(np.uint8)
        return deepcopy(self.image)


class CreateTestImage(ImageOperation):

    def execute(self):
        # create a test image of size 128 x 128
        self.image.data = np.zeros((128, 128), dtype=np.uint8)
        # create a 70 x 70 letter T in the center of the image
        self.image.data[28:49, 28:99] = 255  #top
        self.image.data[48:99, 53:74] = 255  #leg

        return deepcopy(self.image)

class HistogramEqualization(ImageOperation):

    def execute(self):
        # calculate histogram
        histogram = np.histogram(self.image.data, bins=256, range=(0, 255))[0]
        # calculate cumulative histogram
        cumulative_histogram = np.cumsum(histogram)
        # calculate cumulative histogram normalized
        cumulative_histogram_normalized = cumulative_histogram * \
            255 / cumulative_histogram[-1]
        # apply transformation
        self.image.data = cumulative_histogram_normalized[self.image.data]
        return deepcopy(self.image) 
#TODO: refactor affine or pixelwise transformations


class BilinearScaling(ImageOperation):
    '''
    Scaling operation using bilinear interpolation.

    Note: 
        This operation must be configured with a scaling factor.
    '''

    def configure(self, factor):
        self.factor = factor
        return self

    def resize(self, image_data):
        '''Bilinear interpolation'''

        # get the image dimensions
        height, width = np.shape(image_data)

        # get the resize factor
        factor = self.factor

        # calculate the new dimensions
        new_height = round(height * factor)
        new_width = round(width * factor)

        # create a new image with the new dimensions
        new_image = np.zeros((new_height, new_width))

        # get p1, p2, p3 and p4 from original image and then perform bilinear interpolation for each new pixel
        for i in range(new_height):
            for j in range(new_width):
                x = i / factor
                y = j / factor

                new_image[i, j] = bilinear_interp_pixel(x, y, image_data)

        return new_image

    def execute(self):
        self.image.data = self.resize(self.image.data)
        return self.image


class NearestNeighborScaling(ImageOperation):
    '''Scaling operation using nearest neighbor interpolation.
    
    Note:
        This operation must be configured with a scaling factor.
    '''

    def resize(self, image_data):
        # get the image dimensions
        height, width = np.shape(image_data)
        # create a new image with the new dimensions
        new_image = np.zeros((int(round(self.factor * height)),
                              int(round(self.factor * width))))
        # loop through the new image and interpolate the values
        for i in range(0, new_image.shape[0]):  # rows
            for j in range(0, new_image.shape[1]):  # columns
                # get the new image coordinates
                x = i / self.factor
                y = j / self.factor
                # interpolate the pixel value
                new_image[i, j] = nn_interp_pixel(x, y, image_data)
        return new_image

    def execute(self):
        self.image.data = self.resize(self.image.data)
        return self.image


class BilinearRotation(ImageOperation):
    '''Rotation operation using bilinear interpolation.
    
    Note:
        This operation must be configured with a rotation angle.
    '''

    def configure(self, factor):
        self.factor = factor
        return self

    def rotate(self, image_data):
        '''Rotate image using Bilinear interpolation'''
        # get image dimensions
        height, width = np.shape(image_data)

        # create a new image with the new dimensions
        new_image = np.zeros((height, width), dtype=np.uint8)

        # get the center of the image
        center_x = height / 2
        center_y = width / 2

        # get the rotation angle
        angle = self.factor

        # get the rotation matrix
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle),
                                     np.cos(angle)]])
        # get the inverse rotation matrix
        inverse_rotation_matrix = np.linalg.inv(rotation_matrix)

        # iterate over the new image pixels
        for x in range(width):
            for y in range(height):

                # get the pixel coordinates in the original image
                x1 = x - center_x
                y1 = y - center_y

                # apply the inverse rotation matrix
                x2, y2 = inverse_rotation_matrix.dot([x1, y1])

                # get the pixel coordinates in the original image
                x2 = x2 + center_x
                y2 = y2 + center_y

                # interpolate the pixel value
                new_image[x, y] = bilinear_interp_pixel(x2, y2, image_data)

        return new_image

    def execute(self):
        self.image.data = self.rotate(self.image.data)
        return self.image


class NearestNeighborRotation(ImageOperation):
    ''' Rotation operation using nearest neighbor interpolation.

    Note:
        This operation must be configured with a rotation angle.
    '''

    def rotate(self, image_data):
        # get the image dimensions
        height, width = np.shape(image_data)
        # create a new image with the same dimensions
        new_image = np.zeros((height, width), dtype=np.uint8)

        # get the center of the image
        center_x = height / 2
        center_y = width / 2

        # get the rotation angle
        angle = self.factor

        # get the rotation matrix
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle),
                                     np.cos(angle)]])
        # get the inverse rotation matrix
        inverse_rotation_matrix = np.linalg.inv(rotation_matrix)

        # iterate over the new image pixels
        for y in range(width):
            for x in range(height):

                # get the pixel coordinates in the original image
                x1 = x - center_x
                y1 = y - center_y

                # apply the inverse rotation matrix
                x2, y2 = inverse_rotation_matrix.dot([x1, y1])

                # get the pixel coordinates in the original image
                x2 = x2 + center_x
                y2 = y2 + center_y

                # get the nearest pixel
                new_image[x, y] = nn_interp_pixel(x2, y2, image_data)

        return new_image

    def execute(self):
        self.image.data = self.rotate(self.image.data)
        return self.image


class BilinearHorizontalShearing(ImageOperation):
    '''Horizontal shearing operation using bilinear interpolation.
    
    Note:
        This operation must be configured with a shearing factor.
    '''

    def configure(self, factor):
        self.factor = factor
        return self

    def shear(self, image_data):
        '''Shear image using Bilinear interpolation'''
        # get image dimensions
        height, width = np.shape(image_data)

        # create a new image with the new dimensions
        new_image = np.zeros((height, width), dtype=np.uint8)

        # get the shear factor
        shear_factor = -1 * np.tan(np.radians(self.factor) / 2)

        # get the shear matrix
        shear_matrix = np.array([[1, 0], [shear_factor, 1]])

        # get the inverse shear matrix
        inverse_shear_matrix = np.linalg.inv(shear_matrix)

        # get the center of the image
        center_x = height / 2
        center_y = width / 2

        # iterate over the new image pixels
        for y in range(width):
            for x in range(height):
                # move to the origin
                x1 = x - center_x
                y1 = y - center_y

                # apply the inverse shear matrix
                x1, y1 = inverse_shear_matrix.dot([x1, y1])

                # get the pixel coordinates in the original image
                x1 = x1 + center_x
                y1 = y1 + center_y

                # set the new pixel value
                new_image[x, y] = bilinear_interp_pixel(x1, y1, image_data)

        return new_image

    def execute(self):
        self.image.data = self.shear(self.image.data)
        return self.image


class NNHorizontalShearing(ImageOperation):
    ''' Horizontal shearing operation using nearest neighbor interpolation.

    Note:
        This operation must be configured with a shearing factor.
    '''

    def configure(self, factor):
        self.factor = factor
        return self

    def execute(self):
        self.image.data = self.shear(self.image.data)
        return self.image

    def shear(self, image_data):
        # get the image dimensions
        height, width = np.shape(image_data)

        # create a new image with the same dimensions
        new_image = np.zeros((height, width), dtype=np.uint8)

        # get the shear factor
        shear_factor = -1 * np.tan(np.radians(self.factor) / 2)

        # get the shear matrix
        shear_matrix = np.array([[1, 0], [shear_factor, 1]])

        # get the inverse shear matrix
        inverse_shear_matrix = np.linalg.inv(shear_matrix)

        # get the center of the image
        center_x = height / 2
        center_y = width / 2

        # iterate over the new image pixels
        for y in range(width):
            for x in range(height):
                # move to center
                x1 = x - center_x
                y1 = y - center_y

                # apply the inverse shear matrix
                x1, y1 = inverse_shear_matrix.dot([x1, y1])

                # move back to center to get original
                x1 = x1 + center_x
                y1 = y1 + center_y

                # get the nearest pixel
                new_image[x, y] = nn_interp_pixel(x1, y1, image_data)

        return new_image
