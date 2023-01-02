"""Image operations module.
Created on 2022/11
Author: M. Nasser Gaafar

This module contains the operations that can be applied to an image.
Operations are applied to an image by creating an instance of the operation 
configuring that operation instance and then executing it.

Operations are implemented as classes that inherit from the ImageOperation
class. The ImageOperation class is an abstract class that defines the
interface for all operations. The ImageOperation class defines the following
methods:
    configure: configure the operation with the given parameters.
    execute: execute the operation on the image.

Classes:
    ImageOperation: abstract class that defines the interface for all operations.
    MonochoromeConversion: convert the image to monochorome.
    CreateTestImage: create a test image.
    AddSaltPepperNoise: add salt and pepper noise to the image.
    ApplyLinearFilter: apply a linear filter to the image.
    ApplyMedianFilter: apply a median filter to the image.
    HistogramEqualization: apply histogram equalization to the image.
"""

import numpy as np
from modules import utility as util
import threading
from modules.image import *
from modules import interface
from modules.interpolators import *

# TODO: refactor affine or pixelwise transformations
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

    def configure(self, name="t_phantom"):
        self.name = name

    def shepp_logan(self):
        """Create a Schepp-Logan phantom image of size 256 x 256."""
        # create a test image of size 256 x 256
        self.image.data = np.zeros((256, 256), dtype=np.uint8)
        from skimage.data import shepp_logan_phantom
        self.image.data = shepp_logan_phantom()
        self.image.data = (self.image.data * 255).astype(np.uint8)

    def circle_square(self):
        """Create a circle-square phantom image of size 256 x 256."""
        # create a test image of size 256 x 256 with a circle and a square in the center
        self.image.data = np.ones(
            (256, 256), dtype=np.uint8) * 50  # background
        self.image.data[0, 0] = 0  # ground reference point
        # the circle is of radius 50, intensity 50 and the square is of side 160 intensity 150
        self.image.data[48:208, 48:208] = 150  # square
        for x in range(128-50, 128+50):
            for y in range(128-50, 128+50):
                if (x-128)**2 + (y-128)**2 <= 50**2:
                    self.image.data[x, y] = 250  # circle

    def t_phantom(self):
        # create a test image of size 128 x 128
        self.image.data = np.zeros((128, 128), dtype=np.uint8)
        # create a 70 x 70 letter T in the center of the image
        self.image.data[28:49, 28:99] = 255  # top
        self.image.data[48:99, 53:74] = 255  # leg

    def execute(self):

        if self.name == "t_phantom":
            self.t_phantom()
        elif self.name == "shepp_logan":
            self.shepp_logan()
        elif self.name == "circle_square":
            self.circle_square()

        return deepcopy(self.image)


class NoiseGenerator(ImageOperation):

    def configure(self, **kwargs):
        """
        Configure the operation with the given parameters.
        args:
            amount: the percentage of pixels to be affected
            salt_prob: the probability of a pixel to be salted instead of peppered

        """
        if 'type' in kwargs:
            self.type = kwargs['type']
        else:
            raise Exception('Noise type is not specified')

        if self.type == 'salt_pepper':
            self.amount = kwargs['amount']
            self.salt_prob = kwargs['salt_prob']
        elif self.type == 'gaussian':
            self.mean = kwargs['mean']
            self.sigma = kwargs['sigma']
        elif self.type == 'uniform':
            self.a = kwargs['a']
            self.b = kwargs['b']

    def salt_pepper(self):
        # add salt and pepper noise to the image
        # amount is the percentage of pixels to be affected
        # salt_prob is the probability of a pixel to be salted instead of peppered

        L = 2**self.image.get_channel_depth()

        for x in range(self.image.data.shape[0]):
            for y in range(self.image.data.shape[1]):
                if np.random.rand() < self.amount:
                    if np.random.rand() < self.salt_prob:
                        self.image.data[x, y] = L - 1
                    else:
                        self.image.data[x, y] = 0

        return deepcopy(self.image)

    def gaussian(self):
        """ Add gaussian noise to the image """
        L = 2**self.image.get_channel_depth()
        noise = np.random.normal(self.mean, self.sigma, self.image.data.shape)
        self.image.data = self.image.data + noise
        self.image.data = clip(self.image.data, 0, L - 1)
        self.image.data = self.image.data.astype(
            self.image.get_alloc_pixel_dtype())
        return deepcopy(self.image)

    def uniform(self):
        """Add uniform noise to the image"""
        L = 2**self.image.get_channel_depth()
        noise = np.random.uniform(self.a, self.b, self.image.data.shape)
        self.image.data = self.image.data + noise
        self.image.data = clip(self.image.data, 0, L - 1)
        self.image.data = self.image.data.astype(
            self.image.get_alloc_pixel_dtype())
        return deepcopy(self.image)

    def execute(self):
        if self.type == "salt_pepper":
            return self.salt_pepper()
        elif self.type == "gaussian":
            return self.gaussian()
        elif self.type == "uniform":
            return self.uniform()
        else:
            print_debug("Noise error")


class ApplyLinearFilter(ImageOperation):

    def configure(self, **kwargs):
        self.size = kwargs['size']
        self.kernel_type = kwargs['kernel_type']

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
        padded_data = util.uniform_padding(self.image.data, self.size//2, 0)

        if self.kernel_type == 'box':
            self.kernel = self.create_box_kernel()
        else:
            raise ValueError('Unknown kernel type: ' + self.kernel_type)

        # apply a box filter to the image
        # size: size of the filter
        for x in range(self.size, self.image.data.shape[0]+self.size):
            for y in range(self.size, self.image.data.shape[1]+self.size):
                img_section = padded_data[x-self.size:x +
                                          self.size, y-self.size:y+self.size]
                self.image.data[x-self.size, y-self.size] = self.multiply_sum_kernel(
                    self.kernel, img_section)
        # print(f"image data before {self.image.data}")
        data = self.image.data.astype(np.float32)
        # print(f"channel depth {self.image.get_channel_depth()}")
        # print(f"image data {data}")
        self.image.data = np.round(
            data * ((2**self.image.get_channel_depth())-1)/(2**np.log2(np.amax(data))-1))

        return deepcopy(self.image)


class MorphoFilter(ImageOperation):

    def configure(self, **kwargs):
        """Configure the morphological filter
        Args:
            size (int): size of the structuring element
            strel_type (str): type of the structuring element
                options: 'box', 'cross', 'custom'
            operation (str): type of the operation
                options: 'dilation', 'erosion', 'opening', 'closing'
        """
        self.size = kwargs['size']
        self.strel_type = kwargs['strel_type']
        self.operation = kwargs['operation']

    def apply_dilation(self, strel, img_section):
        # apply dilation to the image section
        # kernel: the kernel to be applied
        # img_section: the image section to be filtered
        max = 0
        for x in range(strel.shape[0]):
            for y in range(strel.shape[1]):
                if strel[x, y] == 1 and img_section[x, y] > max:
                    max = img_section[x, y]
        return max

    def apply_erosion(self, strel, img_section):
        # apply erosion to the image section
        # kernel: the kernel to be applied
        # img_section: the image section to be filtered

        min = 255
        for x in range(strel.shape[0]):
            for y in range(strel.shape[1]):
                if strel[x, y] == 1 and img_section[x, y] < min:
                    min = img_section[x, y]
        return min

    def create_strel(self, strel_type, size):
        # create a kernel of size x size
        # strel_type: the type of the kernel
        # size: the size of the kernel
        if strel_type == 'Square':
            strel = np.ones((size, size), dtype=np.float32)
        elif strel_type == 'Cross':
            strel = np.zeros((size, size), dtype=np.float32)
            strel[size//2, :] = 1
            strel[:, size//2] = 1
        elif strel_type == 'Circle':
            strel = np.zeros((size, size), dtype=np.float32)
            for x in range(size):
                for y in range(size):
                    if (x-size//2)**2 + (y-size//2)**2 <= (size//2)**2:
                        strel[x, y] = 1
        elif strel_type == 'Custom':
            # square with zero corners
            strel = np.zeros((size, size), dtype=np.float32)
            strel[0, :] = 1
            strel[size-1, :] = 1
            strel[:, 0] = 1
            strel[:, size-1] = 1

            strel[0, 0] = 0
            strel[0, size-1] = 0
            strel[size-1, 0] = 0
            strel[size-1, size-1] = 0
        else:
            raise ValueError('Unknown kernel type: ' + strel_type)

        return strel

    def loop_image(self, apply_operation,  strel):
        # sync size
        self.size = strel.shape[0]
        # apply operation to the whole image
        original_img = self.image.data.copy()
        padded_img = util.uniform_padding(original_img, self.size//2, 0)
        new_img = padded_img.copy()
        for x in range(self.size//2, padded_img.shape[0]-self.size//2):
            for y in range(self.size//2, padded_img.shape[1]-self.size//2):
                # get section of the image
                img_section = padded_img[x-self.size//2:x+self.size//2 +
                                         1, y-self.size//2:y+self.size//2+1]
                # apply operation to structure element
                new_img[x, y] = apply_operation(strel, img_section)
        # remove padding
        new_img = new_img[self.size//2:new_img.shape[0]-self.size//2,
                          self.size//2:new_img.shape[1]-self.size//2]

        self.image.data = new_img

    def execute(self):
        # apply a morphological filter to the image
        # size: size of the filter
        # kernel_type: the type of the kernel
        # operation: the operation to be applied

        # get kernel
        strel = self.create_strel(self.strel_type, self.size)

        # apply operation
        if self.operation == 'Dilation':
            self.loop_image(self.apply_dilation, strel)
        elif self.operation == 'Erosion':
            self.loop_image(self.apply_erosion, strel)
        elif self.operation == 'Opening':
            # apply erosion
            self.loop_image(self.apply_erosion, strel)
            # apply dilation
            self.loop_image(self.apply_dilation, strel)
        elif self.operation == 'Closing':
            # apply dilation
            self.loop_image(self.apply_dilation, strel)
            # apply erosion
            self.loop_image(self.apply_erosion, strel)
        elif self.operation == 'Fingerprint Cleanup':
            # ignore user input and set kernel to a fingerprint kernel with size 3
            strel = self.create_strel('Square', 3)
            # apply closing
            self.loop_image(self.apply_dilation, strel)
            # apply erosion
            self.loop_image(self.apply_erosion, strel)
            strel = self.create_strel('Cross', 3)
            # apply opening
            self.loop_image(self.apply_erosion, strel)
            # apply dilation
            self.loop_image(self.apply_dilation, strel)
        else:
            raise ValueError('Unknown operation: ' + self.operation)

        return self.image


class BandStopFilter(ImageOperation):
    '''Filters an image by overlaying a radial mask with specified frequency coordinate ranges, has 3 modes:
    Sharp mask: the mask is a sharp edge between the two frequency ranges
    Gaussian mask: the mask is a gaussian function between the two frequency ranges
    Butterworth mask: the mask is a butterworth function between the two frequency ranges
    '''

    def configure(self, **kwargs):
        self.mode = kwargs['mode']
        self.low = kwargs['low']
        self.high = kwargs['high']
        if 'order' in kwargs:
            self.order = kwargs['order']

        if self.mode != 'sharp':
            raise ValueError('Mode is not supported yet')

    def create_sharp_circle_mask(self, low, high):
        # create a mask of circle with raidus low and high and center at the center of the image
        # the mask is 1 inside the circle and 0 outside

        # if low is more than high then swap them and set the mask to 1 outside the circle
        center = (self.image.data.shape[0]//2, self.image.data.shape[1]//2)
        circle_mask = np.ones(self.image.data.shape)
        mask_value = 0
        if low > high:
            low, high = high, low
            circle_mask = np.zeros(self.image.data.shape)
            mask_value = 1
        for x in range(self.image.data.shape[0]):
            for y in range(self.image.data.shape[1]):
                # check if within specified radii
                if (x-center[0])**2 + (y-center[1])**2 <= high**2 and (x-center[0])**2 + (y-center[1])**2 >= low**2:
                    circle_mask[x, y] = mask_value
        return circle_mask

    def execute(self):
        # apply a band stop filter to the image
        # low: low frequency range
        # high: high frequency range
        # mode: sharp, gaussian, butterworth

        if self.low >= self.image.data.shape[0]//2 or self.low >= self.image.data.shape[1]//2:
            raise ValueError('Low frequency range is out of bounds')
        if self.high >= self.image.data.shape[0]//2 or self.high >= self.image.data.shape[1]//2:
            raise ValueError('High frequency range is out of bounds')

        # create mask
        if self.mode == 'sharp':
            mask = self.create_sharp_circle_mask(self.low, self.high)

        # apply mask to image in frequency domain
        image_fft = self.image.get_fft().fft_data
        # shift mask to uncenter it
        mask = np.fft.ifftshift(mask)
        # apply mask
        image_fft = image_fft * mask
        # return image
        self.image.data = np.fft.ifft2(image_fft).real

        return deepcopy(self.image)


class ApplyLinearFilterFreq(ImageOperation):
    '''
    Filters an image in the frequency domain using a linear filter kernel
    '''

    def configure(self, **kwargs):
        self.size = kwargs['size']
        self.kernel_type = kwargs['kernel_type']

    def create_box_kernel(self):
        # create a kernel of size x size with all values = 1 and pad it with image size
        kernel = np.ones((self.size, self.size), dtype=np.float32)/self.size**2
        # normalize the kernel
        kernel = kernel / np.sum(kernel)
        # create zero padding
        kernel = np.zeros(self.image.data.shape)
        # embed the kernel in the center of the padding
        height = self.image.data.shape[0]
        width = self.image.data.shape[1]

        x_offset = (height - self.size) // 2
        y_offset = (width - self.size) // 2

        kernel[x_offset:x_offset + self.size,
               y_offset:y_offset + self.size] = 1

        return kernel

    def apply_kernel_freq(self, kernel):
        # apply the kernel in the frequency domain
        # kernel: the kernel to be applied (padded with image size)

        # get the fourier transform of the image
        img_fft = self.image.get_fft().fft_data

        # inverse shift the fft of the kernel to decenter it
        kernel = np.fft.ifftshift(kernel)
        # get the fourier transform of the kernel
        kernel_fft = np.fft.fft2(kernel)
        # multiply the two fourier transforms
        img_fft = img_fft * kernel_fft
        # get the inverse fourier transform of the result
        img_filtered = np.fft.ifft2(img_fft).real
        return img_filtered

    def execute(self):

        kernel = self.create_box_kernel()

        data = self.apply_kernel_freq(kernel)

        data = np.round(
            data * ((2**self.image.get_channel_depth())-1)/(2**np.log2(np.amax(data))-1))

        self.image.data = data

        return self.image


class ApplyHighboostFilter(ImageOperation):
    '''
    The highboost filter is a linear filter that enhances the edges of an image by subtracting a blurred version of the image from the original image.

    Note: optional argument blur_in_freq changes the blur operation to a frequency domain operation

    '''

    def __post_init__(self):
        self.image2 = deepcopy(self.image)
        self.blur_in_freq = False
        #! would cause a considerable amount of errors
        # TODO: fix pipeline in second image?

    # ? solid principle violation
    def configure(self, **kwargs):
        """
        Configure the operation with the given parameters.

        args:
            size: the size of the filter
            boost: the alpha value of the highboost filter
            clip: whether to clip the image to the range of the original image
        """
        self.boost = kwargs['boost']
        self.clip = kwargs['clip']
        self.size = kwargs['size']
        if 'blur_in_freq' in kwargs:
            self.blur_in_freq = kwargs['blur_in_freq']
        else:
            self.blur_in_freq = False

    def get_sharp_image(self):
        # blur image2
        if self.blur_in_freq:
            linfiltoperation = ApplyLinearFilterFreq()
        else:
            linfiltoperation = ApplyLinearFilter()

        linfiltoperation.configure(size=self.size, kernel_type='box')
        self.image2 = deepcopy(self.image)
        self.image2.clear_operations()
        self.image2.add_operation(MonochoromeConversion())
        self.image2.add_operation(linfiltoperation)
        self.image2.run_processing()
        # diff = np.array([], dtype=np.int)
        # change datatype to int to avoid overflow
        blurred = self.image2.data.astype(np.int)

        difference = self.image.data - blurred

        return difference

    def execute(self):

        L = 2**self.image.get_channel_depth()

        # apply a highboost filter to the image
        self.image.data = np.round(
            self.image.data + self.boost * self.get_sharp_image())

        # normalize or crop
        if self.clip:
            # clip values to [0, L]
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

        print_log('Highboost filter applied')
        print_log('Max Level: ' + str(np.max(self.image.data)))
        print_log('Data Type: ' + str(self.image.data.dtype))
        print_log('Clipping enable: ' + str(self.clip))

        return deepcopy(self.image)


class ApplyMedianFilter(ImageOperation):

    def configure(self, **kwargs):
        """
        Configure the operation with the given parameters.
        args:
            size: the size of the filter
        """
        self.size = kwargs['size']
        # self.maintain_padding = kwargs['maintain_padding']

    def execute(self):
        # apply a median filter to the image
        # size: size of the filter
        # zero padding
        padded_image = util.uniform_padding(self.image.data, self.size//2, 0)
        out_image = self.image.data

        for x in range(self.size, self.image.data.shape[0] + self.size):
            for y in range(self.size, self.image.data.shape[1] + self.size):
                img_section = padded_image[x-self.size:x +
                                           self.size, y-self.size:y+self.size]

                out_image[x-self.size, y-self.size] = np.median(img_section)

        self.image.data = out_image

        return deepcopy(self.image)

# TODO: use memoization for histogram to save processing time


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
        self.image.data = self.image.data.astype(
            self.image.get_alloc_pixel_dtype())

        return deepcopy(self.image)


# TODO: refactor affine or pixelwise transformations

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
