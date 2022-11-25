import numpy as np
from modules import utility as util
import threading
from modules.image import *
from modules import interface
from modules.interpolators import *

#TODO: refactor affine or pixelwise transformations
# class PerPixelOperation(ImageOperation):
#     def __init__(self, name, image, function, *args, **kwargs):
#         super().__init__(name, image)
#         self.function = function
#         self.args = args
#         self.kwargs = kwargs

#     def execute(self):
#         self.image = self.function(self.image, *self.args, **self.kwargs)


# class AffineOperation(ImageOperation):
#     def __init__(self, name, image, function, *args, **kwargs):
#         super().__init__(name, image)
#         self.function = function
#         self.args = args
#         self.kwargs = kwargs

#     def execute(self):
#         self.image = self.function(self.image, *self.args, **self.kwargs)

class FFTMagnitude(ImageOperation):
    def __init__(self, name, image):
        super().__init__(name, image)

    def execute(self):
        #TODO: implement shifting and log scaling manually
        self.image = np.absolute(np.fft.fftshift(np.fft.fft2(self.image)))

class MonochoromeConversion(ImageOperation):

    def execute(self):
        # calculate mean over image channels (color depth axis = 2)
        if self.image.data.ndim == 3:
            self.image.data = np.mean(self.image.data, axis=2)

        # quantize floating point values to integers
        print_log('Quantizing image to grayscale ' +
                  str(self.image.get_alloc_pixel_dtype()))
        self.image.data = self.image.data.astype(
            self.image.get_alloc_pixel_dtype())
        return deepcopy(self.image)


class CreateTestImage(ImageOperation):

    def execute(self):
        # create a test image of size 128 x 128
        self.image.data = np.zeros((128, 128), dtype=np.uint8)
        # create a 70 x 70 letter T in the center of the image
        self.image.data[28:49, 28:99] = 255  #top
        self.image.data[48:99, 53:74] = 255  #leg

        return deepcopy(self.image)


class AddSaltPepperNoise(ImageOperation):

    def __init__(self, name, image, amount, salt_prob=0.5):
        super().__init__(name, image)
        self.amount = amount
        self.salt_prob = salt_prob

    def execute(self):
        # add salt and pepper noise to the image
        # amount is the percentage of pixels to be affected
        # salt_prob is the probability of a pixel to be salted instead of peppered
        for x in range(self.image.data.shape[0]):
            for y in range(self.image.data.shape[1]):
                if np.random.rand() < self.amount:
                    if np.random.rand() < self.salt_prob:
                        self.image.data[x, y] = 2**self.get_channel_depth()
                    else:
                        self.image.data[x, y] = 0


class ApplyLinearFilter(ImageOperation):

    def __init__(self, name, image, size):
        super().__init__(name, image)
        self.size = size
        self.kernel = None

    def create_box_kernel(self):
        # create a kernel of size x size with all values = 1
        kernel = np.ones((self.size, self.size), dtype=np.float32)
        # normalize the kernel
        kernel = kernel / np.sum(kernel)
        return kernel

    def multiply_sum_kernel(self, kernel, img_section):
        # multiply the kernel with the image section and sum
        # kernel: the kernel to be applied
        # img_section: the image section to be filtered
        sum = 0
        for x in range(kernel.shape[0]):
            for y in range(kernel.shape[1]):
                sum += kernel[x, y] * img_section[x, y]
        return sum

    def execute(self):
        # add padding to image data arr
        padded_data = util.uniform_padding(image.data, self.size//2, 0)
        self.kernel = self.create_box_kernel(self.size)
        # apply a box filter to the image
        # size: size of the filter
        for x in range(0, self.image.data.shape[0]):
            for y in range(0, self.image.data.shape[1]):
                img_section = padded_data[x-self.size:x+self.size, y-self.size:y+self.size]
                self.image.data[x, y] = self.multiply_sum_kernel(
                    self.create_kernel(), img_section)
        
        return deepcopy(self.image)


class ApplyHighboostFilter(ImageOperation):

    def __init__(self, name, image, size, boost):
        super().__init__(name, image, size)
        self.boost = boost
        self.clip = None
        self.image2 = deepcopy(self.image)

        #! would cause a considerable amount of errors
        #TODO: think of cutting the pipeline short
    def get_sharp_image(self):
        #blur image2
        self.image2.add_operation(ApplyLinearFilter(self.size))
        self.image2.run_operations()
        diff = []
        diff.np.astype(np.int)
        diff = self.image2.data - self.image.data
        return diff

    def execute(self):

        # apply a highboost filter to the image
        self.image.data = self.image.data + self.boost * self.get_sharp_image()
        # normalize or crop
        L = 2**self.image.get_channel_depth()

        if self.clip:
            #clip values to [0, L]
            for x in range(0, self.image.data.shape[0]):
                for y in range(0, self.image.data.shape[1]):
                    if self.image.data[x, y] > L-1:
                        self.image.data[x, y] = L-1
                    elif self.image.data[x, y] < 0:
                        self.image.data[x, y] = 0
        else:
            # normalize
            self.image.data = self.image.data - np.min(self.image.data)
            self.image.data = self.image.data / np.max(self.image.data) * (L-1)

        return deepcopy(self.image)

class ApplyMedianFilter(ImageOperation):

    def __init__(self, name, image, size):
        super().__init__(name, image)
        self.size = size

    def execute(self):
        # apply a median filter to the image
        # size: size of the filter

        for x in range(self.image.data.shape[0]):
            for y in range(self.image.data.shape[1]):
                # get the image section
                img_section = self.image.data[x:x + self.size, y:y + self.size]
                # calculate the median
                median = util.get_median(img_section)
                # set the pixel value to the median
                self.image.data[x, y] = median


#TODO: use memoization for histogram to save processing time

class HistogramEqualization(ImageOperation):

    def execute(self):
        # get the cumulative sum of the histogram
        histogram, range_histo = self.image.get_histogram()
        L = range_histo[1] + 1 

        # cumulative sum of histogram array
        cdf = np.zeros(range_histo[1]+1)
        for i in range(len(histogram)):
            for j in range(i):
                cdf[i] += histogram[j]


        # apply the cdf to the image
        height, width = np.shape(self.image.data)

        # normalize the cdf
        for x in range(height):
            for y in range(width):
                self.image.data[x, y] = np.round(cdf[self.image.data[x, y]] *
                                                 (L - 1))

        # quantize to previous integer values
        self.image.data = self.image.data.astype(self.image.get_alloc_pixel_dtype())

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
                                    [np.sin(angle), np.cos(angle)]])
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
