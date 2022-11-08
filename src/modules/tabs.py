from pickle import TRUE
from PyQt5.QtWidgets import QMessageBox
from modules import interface
from modules.operations import *

#!Important: missing functionality
# TODO: fix rotation direction (make it anti-clockwise)
# TODO: make sure shearing is correct
# TODO: add axes option to the output image


#TODO: make the functions more generic and reduce code repetiton
def generate_test_image(self):
    '''Generates a test image'''

    if self.image1 == None:
        self.image1 = Image(data=np.zeros((256, 256), dtype=np.uint8))

    self.image1.clear_image_data()

    self.image1.add_operation(CreateTestImage())

    # run the processing and display the result
    interface.display_run_processing(self)

    # clear the operations
    self.image1.clear_operations(clear_backup=True)


def apply_resize(self):
    '''Applies the resize operation to the image'''
    try:
        # get user input parameters data
        factor = interface.get_user_input(self)['resize factor']

        # get the selected interpolator class
        operation = read_transformation(
            interface.get_user_input(self)['transformation type'],
            interface.get_user_input(self)['interpolation method'])

        if operation == None:
            return

        # configure the resize operation object
        resize_operation = operation.configure(factor)

        apply_image_operation(self, resize_operation)
    except:
        QMessageBox.critical(self, 'Error', 'Error Running Operation')


def apply_rotate(self):
    '''Applies the rotation operation to the image'''
    try:
        # get user input parameters data
        factor = np.radians(interface.get_user_input(self)['rotation angle'])

        # get the selected class
        operation = read_transformation(
            interface.get_user_input(self)['transformation type'],
            interface.get_user_input(self)['interpolation method'])

        if operation == None:
            return

        # configure the resize operation object
        rotate_operation = operation.configure(factor)

        apply_image_operation(self, rotate_operation)
    except:
        QMessageBox.critical(self, 'Error', 'Error Running Operation')


def apply_shear(self):
    '''Applies the shear operation to the image'''
    try:
        # get user input parameters data
        factor = interface.get_user_input(self)['shearing factor']

        # get the selected interpolator class
        operation = read_transformation(
            interface.get_user_input(self)['transformation type'],
            interface.get_user_input(self)['interpolation method'])

        if operation == None:
            return

        # configure the resize operation object
        shear_operation = operation.configure(factor)

        apply_image_operation(self, shear_operation)
    except:
        QMessageBox.critical(self, 'Error', 'Error Running Operation')


def apply_image_operation(self, operation, window=None):
    '''Applies the given operation to the image'''
    try:

        # clear previous operations
        self.image1.clear_operations(clear_backup=False, undo_old=True)

        interface.update_img_resize_dimensions(self, "original",
                                               self.image1.get_pixels())

        # add the operation to the image
        self.image1.add_operation(MonochoromeConversion())
        self.image1.add_operation(operation)

        # run the processing
        if window == None:
            selected_window_idx = int(
                self.output_window_combobox.currentIndex())
        else:
            selected_window_idx = window

        interface.display_run_processing(self, selected_window_idx)
    except:
        QMessageBox.critical(self, 'Error', 'Error Running Operation')


def read_transformation(transformation_name,
                        interpolator_name) -> ImageOperation:
    # array of supported interpolators
    transformation = {
        'Rotate': {
            'Nearest-Neighbor': NearestNeighborRotation(),
            'Bilinear': BilinearRotation(),
        },
        'Shear': {
            'Nearest-Neighbor': NNHorizontalShearing(),
            'Bilinear': BilinearHorizontalShearing(),
        },
        'Resize': {
            'Nearest-Neighbor': NearestNeighborScaling(),
            'Bilinear': BilinearScaling(),
        }
    }
    print_debug("Transformation: " + transformation_name)
    print_debug("Interpolator: " + interpolator_name)

    if interpolator_name == 'None' or transformation_name == 'None':
        return

    if transformation_name in transformation:
        if transformation_name == 'Rotate':
            return transformation['Rotate'][interpolator_name]
        elif transformation_name == 'Shear':
            return transformation['Shear'][interpolator_name]
        elif transformation_name == 'Resize':
            return transformation['Resize'][interpolator_name]
        else:
            raise Warning("Unsupported interpolator")


def apply_histogram(self):
    '''Applies the histogram operation to the image'''
    try:
        # get user input parameters data
        output_original = interface.get_user_input(
            self)['histogram output original']
        output_equalized = interface.get_user_input(
            self)['histogram output equalized']
        output_original_plot = interface.get_user_input(
            self)['histogram output original plot']
        output_equalized_plot = interface.get_user_input(
            self)['histogram output equalized plot']

        # output the original image
        self.image1.add_operation(MonochoromeConversion())
        interface.display_run_processing(self, output_original)
        # output the original image histogram
        histogram, range = self.image1.get_histogram(relative=True)
        interface.display_histogram(self, histogram, range,
                                    output_original_plot)

        # equalize the image
        apply_image_operation(self, HistogramEqualization(), output_equalized)
        histogram, range = self.image1.get_histogram(relative=True)
        interface.display_histogram(self, histogram, range,
                                    output_equalized_plot)


    except:
        QMessageBox.critical(self, 'Error', 'Error Running Operation')