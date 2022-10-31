from PyQt5.QtWidgets import QMessageBox
from modules import interface
from modules.transformations import *
from modules.image import *


# connected to the apply button in resize tab
def resize_image(self):
    '''Resizes the image to the specified dimensions'''
    try:
        # get user input parameters data
        factor = interface.get_user_input(self)['resize factor']

        # get the selected interpolator class
        interpolator = read_resize(
            interface.get_user_input(self)['interpolation method'])

        if interpolator == None:
            return

        # configure the resize operation object
        resize_operation = interpolator.configure(factor)

        # undo previous operations
        self.image1.clear_operations()

        interface.update_img_resize_dimensions(self, "original",
                                               self.image1.get_pixels())

        # add the operation to the image
        self.image1.add_operation(MonochoromeConversion())
        self.image1.add_operation(resize_operation)

        interface.print_statusbar(self, 'Processing Image..')
        # run the processing
        selected_window_idx = int(
            self.interpolate_output_combobox.currentIndex())
        interface.display_run_processing(self, selected_window_idx)

    except:
        QMessageBox.critical(self, 'Error', 'Error Running Operation')
        return


def read_resize(interpolator_name) -> ImageOperation:
    # array of supported interpolators
    interpolators = {
        'Nearest-Neighbor': NearestNeighborScaling(),
        'Bilinear': BilinearScaling(),
        'None': None
    }
    if interpolator_name in interpolators:
        return interpolators[interpolator_name]
    else:
        raise Warning("Unsupported interpolator")


# connected to the apply button in transform tab
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
            
        if self.image1.data.empty:
            QMessageBox.critical(self, 'Error',
                                 'Error Running Operation: No Image Loaded')
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

        selected_window_idx = int(
            self.interpolate_output_combobox.currentIndex())
        interface.display_run_processing(self, selected_window_idx)

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