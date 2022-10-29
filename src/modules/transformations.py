import numpy as np
import threading
from modules.image import *
from modules import interface


def transform_image(self):
    try:

        # get user input parameters data
        factor = interface.get_user_input(self)['resize factor']
        type = interface.get_user_input(self)['transformation type']
        interpolator = interface.get_user_input(
            self)['transformation interpolation method']

        # get the selected interpolator class
        transformation = read_transformation(type, interpolator)

        if transformation == None:
            return

        # configure the transformation operation object
        transformation_operation = transformation.configure(factor)

        # undo previous operations
        self.image1.clear_operations()

        self.image1.add_operation(CreateTestImage())
        interface.update_img_resize_dimensions(self, "original",
                                               self.image1.get_pixels())

        # add the operation to the image
        self.image1.add_operation(transformation_operation)

        interface.print_statusbar(self, 'Processing Image..')
        # run the processing
        self.image1.run_processing()

        # print procesing time in status bar
        str_done = "Done processing in " + \
            str(self.image1.get_processing_time()) + "ms"

        interface.print_statusbar(self, str_done)

        # refresh the display
        selected_window = int(self.interpolate_output_combobox.currentIndex())
        interface.display_pixmap(self,
                                 image=self.image1,
                                 window_index=selected_window)
    except:
        QMessageBox.critical(self, 'Error', 'Error Running Operation')
        return


def read_transformation(interpolator_name,
                        transformation_name) -> ImageOperation:
    # array of supported interpolators
    transformation = {
        'Rotation': {
            'Nearest-Neighbor': NearestNeighborRotation(),
            'Bilinear': BilinearRotation(),
            'None': None
        },
        'Shearing': {
            'Nearest-Neighbor': NNHorizontalShearing(),
            'Bilinear': BilinearHorizontalShearing(),
            'None': None
        }
    }
    if interpolator_name in transformation and transformation_name in transformation:
        if transformation_name == 'Rotation':
            return transformation['Rotation'][interpolator_name]
        elif transformation_name == 'Shearing':
            return transformation['Shearing'][interpolator_name]
    else:
        raise Warning("Unsupported interpolator")


class BilinearRotation(ImageOperation):

    def configure(self, factor):
        self.factor = factor
        return self

    def linear_interp(self, p1, p2, px):
        return p1 * (1 - px) + p2 * px

    def rotate(self, image_data):
        '''Rotate image using Bilinear interpolation'''
        # get image dimensions
        height, width = image_data.shape

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
                y2, x2 = inverse_rotation_matrix.dot([y1, x1])

                # get the pixel coordinates in the original image
                x2 = x2 + center_x
                y2 = y2 + center_y

                # get the nearest 4 pixels
                p1 = image_data[int(x2), int(y2)]  #top left
                p2 = image_data[int(x2), int(y2 + 1)]  #top right
                p3 = image_data[int(x2 + 1), int(y2)]  #bottom left
                p4 = image_data[int(x2 + 1), int(y2 + 1)]  #bottom right

                # get the fractional part of the pixel coordinates
                x2_frac = x2 - int(x2)
                y2_frac = y2 - int(y2)

                # interpolate the pixel intensity values
                p12 = self.linear_interp(p1, p2, y2_frac)
                p34 = self.linear_interp(p3, p4, y2_frac)
                p1234 = self.linear_interp(p12, p34, x2_frac)

                # set the new pixel value
                new_image[x, y] = p1234

        return new_image

    def execute(self):
        self.image.data = self.rotate(self.image.data)
        return self.image


class NearestNeighborRotation(ImageOperation):

    def rotate(self, image_data):
        # get the image dimensions
        height, width = image_data.shape
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
                y2, x2 = inverse_rotation_matrix.dot([y1, x1])

                # get the nearest pixel
                new_image[x, y] = image_data[round(x2), round(y2)]

        return new_image

    def execute(self):
        self.image.data = self.rotate(self.image.data)
        return self.image


class BilinearHorizontalShearing(ImageOperation):

    def configure(self, factor):
        self.factor = factor
        return self

    def linear_interp(self, p1, p2, px):
        return p1 * (1 - px) + p2 * px

    def shear(self, image_data):
        '''Shear image using Bilinear interpolation'''
        # get image dimensions
        height, width = image_data.shape

        # create a new image with the new dimensions
        new_image = np.zeros((height, width), dtype=np.uint8)

        # get the shear factor
        shear_factor = self.factor

        # get the shear matrix
        shear_matrix = np.array([[1, shear_factor], [0, 1]])

        # get the inverse shear matrix
        inverse_shear_matrix = np.linalg.inv(shear_matrix)

        # iterate over the new image pixels
        for x in range(width):
            for y in range(height):

                # get the pixel coordinates in the original image
                x1, y1 = inverse_shear_matrix.dot([x, y])

                # get the nearest 4 pixels
                p1 = image_data[int(x1), int(y1)]  #top left
                p2 = image_data[int(x1), int(y1 + 1)]  #top right
                p3 = image_data[int(x1 + 1), int(y1)]  #bottom left
                p4 = image_data[int(x1 + 1), int(y1 + 1)]  #bottom right

                # get the fractional part of the pixel coordinates
                x1_frac = x1 - int(x1)
                y1_frac = y1 - int(y1)

                # interpolate the pixel intensity values
                p12 = self.linear_interp(p1, p2, y1_frac)
                p34 = self.linear_interp(p3, p4, y1_frac)
                p1234 = self.linear_interp(p12, p34, x1_frac)

                # set the new pixel value
                new_image[x, y] = p1234

        return new_image

    def execute(self):
        self.image.data = self.shear(self.image.data)
        return self.image


class NNHorizontalShearing(ImageOperation):

    def configure(self, factor):
        self.factor = factor
        return self

    def execute(self):
        self.image.data = self.shear(self.image.data)
        return self.image

    def shear(self, image_data):
        # get the image dimensions
        height, width = image_data.shape

        # create a new image with the same dimensions
        new_image = np.zeros((height, width), dtype=np.uint8)

        # get the shear factor
        shear_factor = self.factor

        # get the shear matrix
        shear_matrix = np.array([[1, shear_factor], [0, 1]])

        # get the inverse shear matrix
        inverse_shear_matrix = np.linalg.inv(shear_matrix)

        # iterate over the new image pixels
        for y in range(width):
            for x in range(height):

                # get the pixel coordinates in the original image
                x1, y1 = inverse_shear_matrix.dot([x, y])

                # get the nearest pixel
                new_image[x, y] = image_data[round(x1), round(y1)]

        return new_image
