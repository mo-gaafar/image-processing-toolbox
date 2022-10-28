import numpy as np
import threading
from modules.image import *
from modules import interface


# connected to the apply button in resize tab


def reset_image(self):
    '''Resets the image to its original size'''
    try:
        # undo previous operations
        self.image1.clear_operations()
        selected_window = int(self.interpolate_output_combobox.currentIndex())
        interface.display_pixmap(
            self, image=self.image1, window_index=selected_window)
        interface.update_img_resize_dimensions(
            self, 'resized', self.image1.get_pixels())
        interface.print_statusbar(self, 'Image Reset')

    except:
        QMessageBox.critical(
            self, 'Error', 'Error Running Operation: No Image Loaded')
        return
    # refresh the display


def resize_image(self):
    '''Resizes the image to the specified dimensions'''
    try:
        # get user input parameters data
        factor = interface.get_user_input(self)['resize factor']

        # get the selected interpolator class
        interpolator = read_interpolator(
            interface.get_user_input(self)['interpolation method'])

        if interpolator == None:
            return

        # configure the resize operation object
        resize_operation = interpolator.configure(factor)

        # undo previous operations
        self.image1.clear_operations()

        interface.update_img_resize_dimensions(
            self, "original", self.image1.get_pixels())

        # add the operation to the image
        self.image1.add_operation(MonochoromeConversion())
        self.image1.add_operation(resize_operation)

        interface.print_statusbar(self, 'Processing Image..')
        # run the processing
        self.image1.run_processing()

        # print procesing time in status bar
        str_done = "Done processing in " + \
            str(self.image1.get_processing_time()) + "ms"

        interface.print_statusbar(self, str_done)

        interface.update_img_resize_dimensions(
            self, "resized", self.image1.get_pixels())

        # refresh the display
        selected_window = int(self.interpolate_output_combobox.currentIndex())
        interface.display_pixmap(
            self, image=self.image1, window_index=selected_window)

    except:
        QMessageBox.critical(
            self, 'Error', 'Error Running Operation')
        return


def read_interpolator(interpolator_name) -> ImageOperation:
    # array of supported interpolators
    interpolators = {
        'Nearest-Neighbor': NearestNeighborInterpolator(),
        'Bilinear': BilinearInterpolator(),
        'None': None
    }
    if interpolator_name in interpolators:
        return interpolators[interpolator_name]
    else:
        raise Warning("Unsupported interpolator")


class BilinearInterpolator(ImageOperation):
    def configure(self, factor):
        self.factor = factor
        return self

    def linear_interp(self, p1, p2, px):
        return p1*(1-px) + p2 * px

    def interpolate(self, image_data):
        '''Bilinear interpolation'''

        # get the image dimensions
        height, width = image_data.shape

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
                x = i/factor
                y = j/factor

                x1 = int(np.floor(x))
                x2 = int(np.ceil(x))
                y1 = int(np.floor(y))
                y2 = int(np.ceil(y))

                # p1 -- p' ---- p2
                # |     |       |
                # |     |       |
                # |     |       |
                # p3 --p''---- p4

                # check if p1,p2,p3,p4 are out of bounds
                if x1 < 0 or x2 >= height or y1 < 0 or y2 >= width:
                    if x2 >= height:
                        x2 = x1
                    if y2 >= width:
                        y2 = y1

                # # note: the the x y coordinates were swapped when parsing the image for some reason
                # # x is rows, y is columns in the image
                # # axis 0 -> rows

                # p1 = image_data[y1, x1]
                # p2 = image_data[y1, x2]
                # p3 = image_data[y2, x1]
                # p4 = image_data[y2, x2]

                p1 = image_data[x1, y1]
                p2 = image_data[x1, y2]
                p3 = image_data[x2, y1]
                p4 = image_data[x2, y2]

                # calculate the new pixel value
                new_image[i, j] = self.linear_interp(
                    self.linear_interp(p1, p2, y-y1), self.linear_interp(p3, p4, y-y1), x-x1)
        return new_image

    def execute(self):
        self.image.data = self.interpolate(self.image.data)
        return self.image


class NearestNeighborInterpolator(ImageOperation):

    def interpolate(self, image_data):
        # get the image dimensions
        height, width = image_data.shape
        # create a new image with the new dimensions
        new_image = np.zeros(
            (int(np.round(self.factor * height)), int(np.round(self.factor * width))))
        # loop through the new image and interpolate the values
        for i in range(0, new_image.shape[0]):  # rows
            for j in range(0, new_image.shape[1]):  # columns
                # get the new image coordinates
                x = i / self.factor
                y = j / self.factor
                # get the coordinates of the nearest neighbors
                x1 = int(np.floor(x))
                y1 = int(np.floor(y))
                # get the pixel values of the nearest neighbors
                q11 = image_data[x1, y1]
                # interpolate the pixel value
                new_image[i, j] = q11
        return new_image

    def execute(self):
        self.image.data = self.interpolate(self.image.data)
        return self.image
