"""This module contains the code that manages the tabs in the main window
Created on 2022/11
Author: M. Nasser Gaafar

Functions:
    generate_test_image(self): Generates a test image
    apply_resize(self): Applies the resize operation to the image
    apply_rotate(self): Applies the rotation operation to the image
    apply_shear(self): Applies the shear operation to the image
    apply_image_operation(self, operation): Applies the given operation to the image
    read_transformation(transformation_type, interpolation_method): Reads the transformation type and interpolation method and returns the corresponding class
    apply_histogram(self): Applies the histogram operation to the image
    apply_boxblur(self): Applies the box blur operation to the image
    apply_highboost(self): Applies the high boost operation to the image
    apply_median(self): Applies the order statistic median filter operation to the image
    apply_saltpepper(self): Applies the salt and pepper noise operation to the image
    display_fft(self): Displays the FFT of the image
    apply_ft_blur(self): Applies the frequency domain blur operation to the image
    apply_bandstop(self): Applies the bandstop filter operation to the image

"""

from PyQt5.QtWidgets import QMessageBox
from modules import interface
from modules.operations import *
from modules.image import UpdateFFT

# TODO: make the functions more generic and reduce code repetiton


def generate_test_image(self, name='t_phantom'):
    '''Generates a test image'''

    if self.image1 == None:
        self.image1 = Image(data=np.zeros((256, 256), dtype=np.uint8))

    self.image1.clear_image_data()

    operation = CreateTestImage()
    operation.configure(name=name)

    self.image1.add_operation(operation)

    # run the processing and display the result
    interface.display_run_processing(self)

    self.safe_image_backup = deepcopy(self.image1)
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

        # clear previous operations
        self.image1.clear_operations(clear_backup=False, undo_old=True)

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


def apply_boxblur(self):
    """ Applies the box blur operation to the image """
    try:
        # get uset input data
        size = interface.get_user_input(self)['spfilter kernel size']
        output_filtered = interface.get_user_input(self)['spfilter output']

        # clear previous operations
        # self.image1.clear_operations(clear_backup=False, undo_old=False)

        # add the operation to the image
        self.image1.add_operation(MonochoromeConversion())

        # configure the box blur operation object
        boxblur_operation = ApplyLinearFilter()
        boxblur_operation.configure(size=size, kernel_type='box')

        # add the operation to the image
        self.image1.add_operation(boxblur_operation)

        # run the processing
        interface.display_run_processing(self, output_filtered)

    except:
        QMessageBox.critical(self, 'Error', 'Error Running Operation')


def apply_highboost(self):
    """Applies the highboost operation to the image"""

    try:
        # get uset input data
        size = interface.get_user_input(self)['spfilter kernel size']
        output_filtered = interface.get_user_input(self)['spfilter output']
        factor = interface.get_user_input(self)['spfilter highboost factor']
        clip = interface.get_user_input(self)['spfilter highboost clipping']

        # add the operation to the image

        # configure the highboost operation object
        highboost_operation = ApplyHighboostFilter()
        highboost_operation.configure(size=size, clip=clip,
                                      boost=factor)

        # clear previous operations
        self.image1.clear_operations(clear_backup=True, undo_old=True)

        # add the operation to the image
        self.image1.add_operation(MonochoromeConversion())
        self.image1.add_operation(highboost_operation)

        # run the processing
        interface.display_run_processing(self, output_filtered)

    except:
        QMessageBox.critical(self, 'Error', 'Error Running Operation')


def apply_median(self):
    """Applies the median filter to the image"""

    try:
        # get user input parameters data
        size = interface.get_user_input(self)['spfilter kernel size']
        output_filtered = interface.get_user_input(
            self)['spfilter output']

        # clear previous operations
        self.image1.clear_operations(clear_backup=True, undo_old=False)

        # add the operation to the image
        self.image1.add_operation(MonochoromeConversion())

        # configure the median filter operation object
        median_filter = ApplyMedianFilter()
        median_filter.configure(size=size)

        # add the operation to the image
        self.image1.add_operation(median_filter)

        # run the processing
        interface.display_run_processing(self, output_filtered)

    except:
        QMessageBox.critical(self, 'Error', 'Error Running Operation')


def display_fft(self):
    """ Displays the fft of the image """
    try:
        # get user input parameters data
        fftmagshift_window = interface.get_user_input(
            self)['fft output magshift']
        fftmaglog_window = interface.get_user_input(
            self)['fft output maglog']
        fftphaseshift_window = interface.get_user_input(
            self)['fft output phshift']
        fftphaselog_window = interface.get_user_input(
            self)['fft output phlog']

        # clear previous operations
        self.image1.clear_operations(clear_backup=True, undo_old=True)

        # add the operation to the image
        self.image1.add_operation(MonochoromeConversion())

        # configure operation
        operation = UpdateFFT()

        # add the operation to the image
        self.image1.add_operation(operation)

        # run the processing
        self.image1.run_processing()

        # print procesing time in status bar
        str_done = "Done processing in " + \
            str(self.image1.get_processing_time()) + "ms"

        interface.print_statusbar(self, str_done)

        # display the selected outputs

        mag, phase = self.image1.get_fft().process_fft_displays()

        magshift = mag[1]
        maglog = mag[0]

        phshift = phase[1]
        phlog = phase[0]

        if fftmagshift_window != None:
            interface.display_pixmap(self, magshift, fftmagshift_window)
        if fftmaglog_window != None:
            interface.display_pixmap(self, maglog, fftmaglog_window)
        if fftphaseshift_window != None:
            interface.display_pixmap(self, phshift, fftphaseshift_window)
        if fftphaselog_window != None:
            interface.display_pixmap(self, phlog, fftphaselog_window)

    except:
        QMessageBox.critical(self, 'Error', 'Error Running Operation')


def apply_ft_blur(self):
    """Applies the fourier blur operation to the image"""

    try:
        # get user input parameters data
        size = interface.get_user_input(self)['ftfilter kernel size']
        output_filtered = interface.get_user_input(self)['ftfilter output']
        output_spfiltered = interface.get_user_input(
            self)['ftfilter spfilter output']
        comparison_enabled = interface.get_user_input(self)['ftfilter compare']
        diff_window = interface.get_user_input(self)['ftfilter diff output']

        # clear previous operations
        self.image1.clear_operations(clear_backup=True, undo_old=True)
        temp_backup = deepcopy(self.image1)

        # add the operation to the image
        self.image1.add_operation(MonochoromeConversion())

        # configure the fourier blur operation object
        fourier_blur = ApplyLinearFilterFreq()
        fourier_blur.configure(size=size, kernel_type='box')

        # add the operation to the image
        self.image1.add_operation(fourier_blur)

        # run the processing
        interface.display_run_processing(
            self, output_filtered, force_normalize=False)

        if comparison_enabled == True:
            self.image1.clear_operations(clear_backup=True, undo_old=True)

            temp_ftfiltered = deepcopy(self.image1)

            self.image1 = deepcopy(temp_backup)

            self.image1.add_operation(MonochoromeConversion())

            # configure the spatial blur operation object
            spfilter = ApplyLinearFilter()
            spfilter.configure(size=size, kernel_type='box')

            # add the operation to the image
            self.image1.add_operation(spfilter)

            # run the processing
            interface.display_run_processing(
                self, output_spfiltered, force_normalize=False)

            temp_spfiltered = deepcopy(self.image1)

            diff = np.abs(temp_ftfiltered.data - temp_spfiltered.data)

            print("spfiltered")
            print(temp_spfiltered.data)
            print("ftfilered")
            print(temp_ftfiltered.data)
            print("diff")
            print(diff)

            interface.display_pixmap(
                self, diff, diff_window, force_normalize=False)

    except:
        QMessageBox.critical(self, 'Error', 'Error Running Operation')


def apply_bandstop(self):
    """Applies the bandstop filter to the image"""

    try:
        # get user input parameters data
        output_filtered = interface.get_user_input(self)['ftfilter output']
        low = interface.get_user_input(self)['bandstop low']
        high = interface.get_user_input(self)['bandstop high']

        # clear previous operations
        self.image1.clear_operations(clear_backup=True, undo_old=True)

        # add the operation to the image
        self.image1.add_operation(MonochoromeConversion())

        # configure the bandstop operation object
        bandstop_operation = BandStopFilter()
        bandstop_operation.configure(high=high, low=low, mode='sharp')

        # add the operation to the image
        self.image1.add_operation(bandstop_operation)

        # run the processing
        interface.display_run_processing(self, output_filtered)

    except:
        QMessageBox.critical(self, 'Error', 'Error Running Operation')


def apply_noise(self):
    '''Applies the salt and pepper noise operation to the image'''
    try:
        # get user input parameters data
        output_noisy = interface.get_user_input(self)['noise output']
        noise_type = interface.get_user_input(self)['noise type']

        operation = NoiseGenerator()
        # clear previous operations
        self.image1.clear_operations(clear_backup=True, undo_old=False)

        if noise_type == "salt_pepper":
            salt_noise = interface.get_user_input(self)['saltpepper salt prob']
            salt_noise = salt_noise / 100

            noise_weight = interface.get_user_input(
                self)['saltpepper noise weight']
            noise_weight = noise_weight / 100

            # configure operation
            operation.configure(amount=noise_weight,
                                salt_prob=salt_noise, type=noise_type)

        elif noise_type == "gaussian":
            mean = interface.get_user_input(self)['noise gaussian mean']
            sigma = interface.get_user_input(self)['noise gaussian sigma']

            operation.configure(mean=mean, sigma=sigma, type=noise_type)

        elif noise_type == "uniform":
            a = interface.get_user_input(self)['noise uniform a']
            b = interface.get_user_input(self)['noise uniform b']

            operation.configure(a=a, b=b,  type=noise_type)

        # add the noise
        apply_image_operation(self, operation, output_noisy)

    except:
        QMessageBox.critical(self, 'Error', 'Error Running Operation')


def roi_select(self):
    """ Displays the roi selection window """
    try:
        # get user input parameters data
        roi_window = interface.get_user_input(self)['roi select output']
        # display the roi selection window
        interface.show_roi_window(self, roi_window)
    except:
        QMessageBox.critical(self, 'Error', 'No Image Loaded or RGB Image')


def analyze_roi(self):
    try:
        # get user input parameters data
        roi_coordinates = self.selected_roi_coords
        analysis_window = interface.get_user_input(self)['roi analysis output']

        # get a sub region using the roi coordinates
        roi = self.image1.get_region(roi_coordinates)

        # get and display histogram of roi
        histogram = get_histogram(roi)
        histo_range = None
        interface.display_histogram(
            self, histogram, histo_range, analysis_window)

        # get and display mean of roi
        mean = histo_mean(histogram=histogram)
        outstr = ""
        outstr = "Mean: " + str(round(mean, 2))
        interface.display_text(self, outstr, 'noise mean')

        # get and display standard deviation of roi
        std = histo_std_dev(histogram=histogram)
        outstr = ""
        outstr += "Standard Deviation: " + str(round(std, 2))
        interface.display_text(self, outstr, 'noise std dev')
    except:
        QMessageBox.critical(self, 'Error', 'Error Running Operation')


def display_sinogram(self):
    # get range array from user input

    # get display window

    # process sinogram here

    # display sinogram
    pass


def display_laminogram(self):
    # get filter type

    # get display window

    # process laminogram

    # display laminogram

    pass
