"""This module contains the functions that are used to interact with the GUI.
Created on 2022/11
Author: M. Nasser Gaafar

Functions:
    init_sync_sliders(self): initializes the dictionary that contains the sliders and spinboxes that are synced
    sync_slider(self, slider, spinbox): syncs the slider and spinbox
    sync_spinbox(self, slider, spinbox): syncs the spinbox and slider
    update_img_resize_dimensions(self): updates the image resize dimensions
    get_user_input(self): gets the user input from the GUI
    print_statusbar(self,message): prints a message in the statusbar
    display_run_processing(self, selected_window, kwargs): runs the processing function and displays the result
    display_pixmap(self, image, window, force_normalize): displays an image in a window
    display_histogram(self, histogram, range, window): displays a histogram in a window
    toggle_image_window(self, windowindex): toggles the image window hide/show
    save_tab_references(self, tab): saves the references to the tabs
    toggle_toolbox_tabs(self, tab): toggles the toolbox tabs hide/show
    display_metatable(self, f_metadata): displays the metadata table

"""

import numpy as np
from PIL import Image as PILImage
from PyQt5.QtWidgets import QTableWidgetItem, QMessageBox, QToolBox, QWidget, QTabWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pylab as plt
from PyQt5.QtGui import *
from PyQt5 import QtGui
from modules.utility import print_debug, round_nearest_odd
from modules import openfile, operations, tabs
import modules.image


def init_sync_sliders(self):
    self.tab_slider_sync_dict = {
        'resize': {
            'slider': self.resize_slider,
            'spinbox': self.resize_spinbox
        },
        'rotate': {
            'slider': self.rotation_slider,
            'spinbox': self.rotation_angle_spinbox
        }
    }


def sync_sliders(self, caller=None, name=None):
    ''' Synchrnoizes slider values and spinbox values'''
    if name == 'resize':
        if caller == 'slider':
            self.resize_spinbox.setValue(self.resize_slider.value() / 10)
        elif caller == 'spinbox':
            self.resize_slider.setValue(int(self.resize_spinbox.value() * 10))
    elif name == 'rotate':
        if caller == 'slider':
            self.rotation_angle_spinbox.setValue(self.rotation_slider.value() %
                                                 360)
        elif caller == 'spinbox':
            self.rotation_slider.setValue(
                int(self.rotation_angle_spinbox.value()) % 360)
    elif name == 'spfilter kernel size':
        if caller == 'slider':
            self.sp_kernel_spinbox.setValue(
                round_nearest_odd(self.sp_kernel_slider.value()))

        elif caller == 'spinbox':
            self.sp_kernel_spinbox.setValue(
                round_nearest_odd(self.sp_kernel_spinbox.value()))
            self.sp_kernel_slider.setValue(
                int(self.sp_kernel_spinbox.value()))
    elif name == 'spfilter highboost factor':
        if caller == 'slider':
            self.highboost_factor_spinbox.setValue(
                self.highboost_factor_slider.value()/2)
        elif caller == 'spinbox':
            self.highboost_factor_slider.setValue(
                int(self.highboost_factor_spinbox.value()*2))
    elif name == 'saltpepper noise weight':
        if caller == 'slider':
            self.noise_wt_spinbox.setValue(self.noise_wt_slider.value())
        elif caller == 'spinbox':
            self.noise_wt_slider.setValue(
                int(self.noise_wt_spinbox.value()))
    elif name == 'saltpepper salt prob':
        if caller == 'slider':
            self.noise_salt_spinbox.setValue(self.noise_salt_slider.value())
        elif caller == 'spinbox':
            self.noise_salt_slider.setValue(
                int(self.noise_salt_spinbox.value()))
    elif name == 'ftfilter kernel size':
        if caller == 'slider':
            self.ft_kernel_spinbox.setValue(
                round_nearest_odd(self.ft_kernel_slider.value()))

        elif caller == 'spinbox':
            self.ft_kernel_spinbox.setValue(
                round_nearest_odd(self.ft_kernel_spinbox.value()))
            self.ft_kernel_slider.setValue(
                int(self.ft_kernel_spinbox.value()))
    else:
        raise ValueError("Incorrect spinbox or slider name")


def update_img_resize_dimensions(self, selector, data_arr):
    if selector == 'original':
        dimensions_string = str(np.shape(data_arr)[0]) + " x " + str(
            np.shape(data_arr)[1]) + " px"
        self.resize_original_dim_textbox.setText(dimensions_string)
    elif selector == 'resized':
        dimensions_string = str(np.shape(data_arr)[0]) + " x " + str(
            np.shape(data_arr)[1]) + " px"
        self.resize_modified_dim_textbox.setText(dimensions_string)


# global dictionary for output window selection
output_window_dict = {
    'Image1': 0,
    'Image2': 1,
    'Plot1': 2,
    'Plot2': 3,
    'None': None
}


def get_user_input(self):
    '''
    Gets the user input from the GUI and returns it as a dictionary.

    Keys:
        'interpolation method',
        'transformation type',
        'resize factor',
        'rotation angle',
        'shearing factor',
        'output window',
        'histogram output window',
        'histogram output equalized',
        'histogram output original plot',
        'histogram output equalized plot',
        'spfilter output',
        'spfilter kernel size',
        'spfilter highboost factor',
        'spfilter highboost clipping',
        'saltpepper noise weight',
        'saltpepper salt prob'

    '''
    user_input = {}

    # Affine Transformations
    user_input['interpolation method'] = self.interpolate_combobox.currentText(
    )
    user_input['transformation type'] = self.toolbox_transform.itemText(
        self.toolbox_transform.currentIndex())
    user_input['resize factor'] = self.resize_spinbox.value()
    user_input['rotation angle'] = self.rotation_angle_spinbox.value()
    user_input['shearing factor'] = self.shearing_factor_spinbox.value()

    user_input['output window'] = output_window_dict[
        self.output_window_combobox.currentText()]

    # Histogram Equalization
    user_input['histogram output original'] = output_window_dict[
        self.histo_output_original_combobox.currentText()]
    user_input['histogram output equalized'] = output_window_dict[
        self.histo_output_equalized_combobox.currentText()]
    user_input['histogram output original plot'] = output_window_dict[
        self.histo_output_original_plot_combobox.currentText()]
    user_input['histogram output equalized plot'] = output_window_dict[
        self.histo_output_equalized_plot_combobox.currentText()]

    # Spatial Filtering
    user_input['spfilter output'] = output_window_dict[self.spfilter_output_combobox.currentText()]
    user_input['spfilter kernel size'] = int(self.sp_kernel_spinbox.value())

    user_input['spfilter highboost factor'] = self.highboost_factor_spinbox.value()
    user_input['spfilter highboost clipping'] = self.clipping_checkbox.isChecked()

    user_input['saltpepper noise weight'] = self.noise_wt_spinbox.value()
    user_input['saltpepper salt prob'] = self.noise_salt_spinbox.value()

    # FFT Display
    user_input['fft output magshift'] = output_window_dict[self.fft_output_magshift_combobox.currentText()]
    user_input['fft output maglog'] = output_window_dict[self.fft_output_maglog_combobox.currentText()]

    user_input['fft output phshift'] = output_window_dict[self.fft_output_phshift_combobox.currentText()]
    user_input['fft output phlog'] = output_window_dict[self.fft_output_phlog_combobox.currentText()]

    # FT Filtering

    user_input['ftfilter output'] = output_window_dict[self.ftfilter_output_combobox.currentText()]
    user_input['ftfilter kernel size'] = int(self.ft_kernel_spinbox.value())
    user_input['ftfilter compare'] = self.ftfilter_compare_checkbox.isChecked()
    user_input['ftfilter spfilter output'] = output_window_dict[self.ftfilter_spfilter_output_combobox.currentText()]
    user_input['ftfilter diff output'] = output_window_dict[self.ftfilter_diff_output_combobox.currentText()]

    # user_input['bandstop output'] = output_window_dict[self.bandstop_output_combobox.currentText()]

    user_input['bandstop low'] = int(self.bandstop_low_spinbox.value())
    user_input['bandstop high'] = int(self.bandstop_high_spinbox.value())

    return user_input


def print_statusbar(self, message):
    self.statusbar.showMessage(message)


def display_run_processing(self, selected_window_idx=None, **kwargs):

    print_statusbar(self, 'Processing Image..')
    # run the processing
    self.image1.run_processing()

    # print procesing time in status bar
    str_done = "Done processing in " + \
        str(self.image1.get_processing_time()) + "ms"

    print_statusbar(self, str_done)

    # update resize dimensions
    update_img_resize_dimensions(self, 'resized', self.image1.get_pixels())

    # refresh the display
    display_pixmap(self, image=self.image1,
                   window_index=selected_window_idx, **kwargs)


plt.rcParams['axes.facecolor'] = 'black'
plt.rc('axes', edgecolor='white')
plt.rc('xtick', color='white')
plt.rc('ytick', color='white')
plt.rcParams["figure.autolayout"] = True


def display_pixmap(self, image, window_index=None, force_normalize=True):
    '''Displays the image data in the image display area.

    Args:
        image (Image): Image object to be displayed. (or numpy array)
        window_index (int): Index of the window to be displayed in.
    '''
    qim = None

    if type(image) == modules.image.Image:
        image_data = image.get_pixels()
    else:
        image_data = image
    # quantize the image data to 8 bit for display
    if force_normalize or image_data.min() < 0 or image_data.max() > 255:
        image_data = np.interp(image_data,
                               (image_data.min(), image_data.max()), (0, 255))
    image_data = np.array(image_data, dtype=np.uint8)

    print_debug("Displaying Image")
    print_debug(np.shape(image_data))

    if len(np.shape(image_data)) == 2:
        im = PILImage.fromarray(image_data)
        if im.mode != 'L':
            im = im.convert('L')

        qim = im.toqimage()

    elif len(np.shape(image_data)) == 3:
        try:
            im = PILImage.fromarray(image_data)
            if im.mode != 'RGB':
                im = im.convert('RGB')
        except TypeError:
            image_data = image_data.astype(np.uint8)
            image_data = np.mean(image_data, axis=2)
            im = PILImage.fromarray(image_data, 'L')
        except:
            # error message pyqt
            QMessageBox.critical(
                self, 'Error',
                'Error Displaying Image: Unsupported Image Format')
            return

        data = im.tobytes(encoder_name='raw')

        totalbytes = len(data)
        bytesperline = int(totalbytes / im.size[1])

        # fix skewed image qim

        qim = QtGui.QImage(data, im.size[0], im.size[1], bytesperline,
                           QtGui.QImage.Format_RGB888)

    # convert the image to binary in RGB format

    # display saved image in Qpixmap
    if window_index == None:
        window_index = get_user_input(self)['output window']

    if window_index == 0:
        self.image1_pixmap = QPixmap.fromImage(qim)
        self.image1_widget.setPixmap(self.image1_pixmap)
        self.image1_widget.adjustSize()
        self.image1_widget.show()
    elif window_index == 1:
        self.image2_pixmap = QPixmap.fromImage(qim)
        self.image2_widget.setPixmap(self.image2_pixmap)
        self.image2_widget.adjustSize()
        self.image2_widget.show()
    elif window_index == 2:

        self.figure = plt.figure(figsize=(15, 5), facecolor='black')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect('button_press_event',
                                self.output_click_statusbar)
        self.gridLayout_11.addWidget(self.canvas, 0, 0, 1, 1)
        plt.axis('on')
        plt.imshow(image_data, cmap='gray', interpolation=None)
        self.canvas.draw()
    elif window_index == 3:
        self.figure = plt.figure(figsize=(15, 5), facecolor='black')
        self.canvas2 = FigureCanvas(self.figure)
        self.canvas2.mpl_connect('button_press_event',
                                 self.output_click_statusbar)
        self.gridLayout_15.addWidget(self.canvas2, 0, 0, 1, 1)
        plt.axis('on')
        plt.imshow(image_data, cmap='gray', interpolation=None)
        self.canvas2.draw()
    else:
        raise ValueError("Invalid window index")


def display_histogram(self, histogram, range_hist, window_index=None):
    '''Displays the histogram of the image in the histogram display area'''

    if window_index == None:
        return

    if window_index == output_window_dict['Plot1']:
        self.figure = plt.figure(figsize=(15, 5), facecolor='black')
        self.canvas = FigureCanvas(self.figure)
        self.gridLayout_11.addWidget(self.canvas, 0, 0, 1, 1)
        plt.axis('on')
        plt.bar(range(range_hist[0], range_hist[1] + 1),
                histogram,
                color='red')
        self.canvas.draw()
    elif window_index == output_window_dict['Plot2']:
        self.figure = plt.figure(figsize=(15, 5), facecolor='black')
        self.canvas = FigureCanvas(self.figure)
        self.gridLayout_15.addWidget(self.canvas, 0, 0, 1, 1)
        plt.axis('on')
        plt.bar(range(range_hist[0], range_hist[1] + 1),
                histogram,
                color='blue')
        self.canvas.draw()


def toggle_image_window(self, window_index):
    if window_index == 0:
        if self.image1_groupbox.isHidden():
            self.image1_groupbox.show()
            self.actionImage1.setChecked(True)
        else:
            self.image1_groupbox.hide()
            self.actionImage1.setChecked(False)
    elif window_index == 1:
        if self.image2_groupbox.isHidden():
            self.image2_groupbox.show()
            self.actionImage2.setChecked(True)
        else:
            self.image2_groupbox.hide()
            self.actionImage2.setChecked(False)
    elif window_index == 2:
        if self.plot1_groupbox.isHidden():
            self.plot1_groupbox.show()
            self.actionPlot1.setChecked(True)
        else:
            self.plot1_groupbox.hide()
            self.actionPlot1.setChecked(False)
    elif window_index == 3:
        if self.plot2_groupbox.isHidden():
            self.plot2_groupbox.show()
            self.actionPlot2.setChecked(True)
        else:
            self.plot2_groupbox.hide()
            self.actionPlot2.setChecked(False)
    else:
        raise ValueError("Invalid window index")


toolbox_tab_dict = {
    'Metadata': 'tab_meta',
    'Transform': 'tab_transform',
    'Histogram': 'tab_histogram',
    'Spatial Filter': 'tab_spfilt',
    'FFT Display': 'tab_fftdisp',
    'Frequency Filter': 'tab_freqfilt',
}


def save_tab_references(self):
    self.tab_meta = self.tabWidget.findChild(QWidget, 'tab_meta')
    self.tab_transform = self.tabWidget.findChild(QWidget, 'tab_transform')
    self.tab_histogram = self.tabWidget.findChild(QWidget, 'tab_histogram')
    self.tab_spfilt = self.tabWidget.findChild(QWidget, 'tab_spfilt')
    self.tab_fftdisp = self.tabWidget.findChild(QWidget, 'tab_fftdisp')
    self.tab_freqfilt = self.tabWidget.findChild(QWidget, 'tab_freqfilt')

    self.toolbox_tab_ref = {
        'Metadata': self.tab_meta,
        'Transform': self.tab_transform,
        'Histogram': self.tab_histogram,
        'Spatial Filter': self.tab_spfilt,
        'FFT Display': self.tab_fftdisp,
        'Frequency Filter': self.tab_freqfilt,
    }


def toggle_toolbox_tabs(self, tab_name):
    tab = self.findChild(QWidget, toolbox_tab_dict[tab_name])

    if tab.isEnabled() == False:
        # tab is hidden
        # get last tab index
        last_tab_index = int(self.tabWidget.count() - 1)
        self.tabWidget.insertTab(
            last_tab_index, self.toolbox_tab_ref[tab_name], tab_name)
        tab.setEnabled(True)
    else:
        # tab is visible
        # disable tab
        tab.setEnabled(False)
        self.tabWidget.removeTab(self.tabWidget.indexOf(tab))


def display_metatable(self, f_metadata=None):
    '''Displays the metadata in QTableWidget'''
    # f_metadata = self.image1.get_metadata()
    self.metadata_tablewidget.clear()
    self.metadata_tablewidget.setRowCount(len(f_metadata))
    self.metadata_tablewidget.setColumnCount(2)
    self.metadata_tablewidget.setHorizontalHeaderLabels(['Property', 'Value'])
    for i, (key, value) in enumerate(f_metadata.items()):
        self.metadata_tablewidget.setItem(i, 0, QTableWidgetItem(str(key)))
        self.metadata_tablewidget.setItem(i, 1, QTableWidgetItem(str(value)))


def disp_fft_all_none(self):
    ''' Sets all display comboboxes to none'''
    self.fft_output_magshift_combobox.setCurrentIndex(4)
    self.fft_output_maglog_combobox.setCurrentIndex(4)
    self.fft_output_phshift_combobox.setCurrentIndex(4)
    self.fft_output_phlog_combobox.setCurrentIndex(4)


def init_connectors(self):
    '''Initializes all event connectors and triggers'''

    ''' Menu Bar'''

    # File Menu
    self.actionOpen.triggered.connect(lambda: openfile.open_new(self, 1))
    self.actionSave_As.triggered.connect(lambda: openfile.save_file(self))
    self.actionSave.triggered.connect(lambda: openfile.save_file(self))

    # View Menu
    # self.actionMetadata.triggered.connect(lambda: self.toggle_metadata_tab())
    # self.actionResizer.triggered.connect(lambda: self.toggle_resize_tab())
    self.actionImage1.triggered.connect(lambda: toggle_image_window(self, 0))
    self.actionImage2.triggered.connect(lambda: toggle_image_window(self, 1))
    self.actionPlot1.triggered.connect(lambda: toggle_image_window(self, 2))
    self.actionPlot2.triggered.connect(lambda: toggle_image_window(self, 3))

    save_tab_references(self)
    self.actionMetadata.triggered.connect(
        lambda: toggle_toolbox_tabs(self, 'Metadata'))
    self.actionTransform.triggered.connect(
        lambda: toggle_toolbox_tabs(self, 'Transform'))
    self.actionHistogramEQ.triggered.connect(
        lambda: toggle_toolbox_tabs(self, 'Histogram'))
    self.actionSpatial_Filter.triggered.connect(
        lambda: toggle_toolbox_tabs(self, 'Spatial Filter'))
    self.actionFFT_Display.triggered.connect(
        lambda: toggle_toolbox_tabs(self, 'FFT Display'))
    # self.actionFrequency_Filter.triggered.connect(lambda: toggle_toolbox_tabs(self, 'Frequency Filter'))

    # Help Menu
    self.actionAbout.triggered.connect(lambda: about_us(self))

    ''' Transformation Tab'''
    # Resize Tab
    self.resize_slider.sliderReleased.connect(
        lambda: sync_sliders(self, 'slider', 'resize'))
    self.resize_spinbox.valueChanged.connect(
        lambda: sync_sliders(self, 'spinbox', 'resize'))
    # Transform Tab
    init_sync_sliders(self)
    # triggers the resizing
    self.resize_apply.clicked.connect(lambda: tabs.apply_resize(self))
    # triggers the rotation
    self.rotate_apply.clicked.connect(lambda: tabs.apply_rotate(self))
    # triggers the shear
    self.shear_apply.clicked.connect(lambda: tabs.apply_shear(self))
    # undo transformations
    self.reset_operations.clicked.connect(
        lambda: modules.image.reset_image(self))
    # generate test image "T"
    self.gen_test_image.clicked.connect(lambda: tabs.generate_test_image(self))
    print_debug("Connectors Initialized")
    # sync rotation sliders
    self.rotation_angle_spinbox.valueChanged.connect(
        lambda: sync_sliders(self, 'spinbox', 'rotate'))
    self.rotation_slider.sliderReleased.connect(
        lambda: sync_sliders(self, 'slider', 'rotate'))

    """ Histogram Tab"""
    # Histogram Tab
    self.histogram_apply.clicked.connect(lambda: tabs.apply_histogram(self))
    self.reset_operations_2.clicked.connect(
        lambda: modules.image.reset_image(self))

    """ Spatial Filtering Tab """
    # Spatial Filtering Tab
    self.boxblur_apply.clicked.connect(lambda: tabs.apply_boxblur(self))
    self.highboost_apply.clicked.connect(lambda: tabs.apply_highboost(self))
    self.median_apply.clicked.connect(lambda: tabs.apply_median(self))
    self.saltpepper_apply.clicked.connect(lambda: tabs.apply_saltpepper(self))
    self.reset_operations_3.clicked.connect(
        lambda: modules.image.restore_original(self))

    # Sync sliders
    self.sp_kernel_slider.sliderReleased.connect(
        lambda: sync_sliders(self, 'slider', 'spfilter kernel size'))
    self.sp_kernel_spinbox.valueChanged.connect(
        lambda: sync_sliders(self, 'spinbox', 'spfilter kernel size'))

    self.highboost_factor_slider.sliderReleased.connect(
        lambda: sync_sliders(self, 'slider', 'spfilter highboost factor'))
    self.highboost_factor_spinbox.valueChanged.connect(
        lambda: sync_sliders(self, 'spinbox', 'spfilter highboost factor'))

    self.noise_wt_slider.sliderReleased.connect(
        lambda: sync_sliders(self, 'slider', 'saltpepper noise weight'))
    self.noise_wt_spinbox.valueChanged.connect(
        lambda: sync_sliders(self, 'spinbox', 'saltpepper noise weight'))

    self.noise_salt_slider.sliderReleased.connect(
        lambda: sync_sliders(self, 'slider', 'saltpepper salt prob'))
    self.noise_salt_spinbox.valueChanged.connect(
        lambda: sync_sliders(self, 'spinbox', 'saltpepper salt prob'))

    self.ft_kernel_slider.sliderReleased.connect(
        lambda: sync_sliders(self, 'slider', 'ftfilter kernel size'))
    self.ft_kernel_spinbox.valueChanged.connect(
        lambda: sync_sliders(self, 'spinbox', 'ftfilter kernel size'))

    """FFT Display Tab"""

    self.disp_fft_apply.clicked.connect(lambda: tabs.display_fft(self))
    self.disp_fft_allnone.clicked.connect(lambda: disp_fft_all_none(self))

    """ FFT Filtering Tab"""
    self.boxblur_ft_apply.clicked.connect(lambda: tabs.apply_ft_blur(self))
    self.bandstop_ft_apply.clicked.connect(lambda: tabs.apply_bandstop(self))
    self.reset_operations_ftfilt.clicked.connect(
        lambda: modules.image.restore_original(self))


def about_us(self):
    QMessageBox.about(
        self, ' About ',
        'This is a Medical Image Toolbox \nCreated by Senior students from the faculty of Engineering, Cairo Uniersity, Systems and Biomedical Engineering department \n \n Created By: Mohamed Nasser '
    )
