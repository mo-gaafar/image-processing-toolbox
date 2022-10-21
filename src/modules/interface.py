
import numpy as np
from PIL import Image as PILImage
from PyQt5.QtWidgets import QTableWidgetItem, QMessageBox
from PyQt5.QtGui import *
from PyQt5 import QtGui
from modules.utility import print_debug
from modules import openfile
from modules import interpolators


def update_resize_tab(self, caller=None):
    ''' Synchrnoizes slider values and spinbox values'''
    if caller == 'slider':
        self.resize_spinbox.setValue(self.resize_slider.value()/10)
    elif caller == 'spinbox':
        self.resize_slider.setValue(int(self.resize_spinbox.value()*10))


def update_img_resize_dimensions(self, selector, data_arr):
    if selector == 'original':
        dimensions_string = str(
            data_arr.shape[0]) + " x " + str(data_arr.shape[1]) + " px"
        self.resize_original_dim_textbox.setText(dimensions_string)
    elif selector == 'resized':
        dimensions_string = str(
            data_arr.shape[0]) + " x " + str(data_arr.shape[1]) + " px"
        self.resize_modified_dim_textbox.setText(dimensions_string)


def get_user_input(self):
    '''Gets the user input from the GUI and returns it as a dictionary'''
    user_input = {}
    user_input['resize factor'] = self.resize_spinbox.value()
    user_input['interpolation method'] = self.interpolate_combobox.currentText()
    return user_input


# def update_browse(self):
#     ''' Updates the user interface with the current image and data'''
#     try:
#         display_pixmap(self, image=self.image1)
#     except TypeError:
#         # error message pyqt
#         QMessageBox.critical(
#             self, 'Error', 'Error Displaying Image: Unsupported Image Format')

#     display_metatable(self)


def print_statusbar(self, message):
    self.statusbar.showMessage(message)


def display_pixmap(self, image, window_index=0):
    '''Displays the image data in the image display area'''
    # then convert it to image format
    image_data = image.get_pixels()

    print_debug("Displaying Image")
    print_debug(image_data.shape)

    qim = None

    if len(image_data.shape) == 2:
        im = PILImage.fromarray(image_data)
        if im.mode != 'L':
            im = im.convert('L')

        qim = im.toqimage()

    elif len(image_data.shape) == 3:
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
                self, 'Error', 'Error Displaying Image: Unsupported Image Format')
            return

        data = im.tobytes()
        qim = QtGui.QImage(
            data, im.size[0], im.size[1], QtGui.QImage.Format_RGB888)

    # convert the image to binary in RGB format

    # display saved image in Qpixmap
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
    else:
        raise ValueError("Invalid window index")


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
    else:
        raise ValueError("Invalid window index")


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

    # Help Menu
    self.actionAbout.triggered.connect(lambda: about_us(self))

    # self.actionAbout_Us = self.findChild(QAction, "actionAbout_Us")
    # self.actionAbout_Us.triggered.connect(
    #     lambda: about_us(self))

    # self.WindowTabs = self.findChild(QTabWidget, "WindowTabs")

    # ''' browse buttons'''
    # # the index argument maps each function to its respective slot
    # self.insert_image1_pushButton.clicked.connect(
    #     lambda: openfile.open_new(self, 1))

    ''' Interpolation (resize) tab'''
    self.resize_slider.sliderReleased.connect(
        lambda: update_resize_tab(self, 'slider'))
    self.resize_spinbox.valueChanged.connect(
        lambda: update_resize_tab(self, 'spinbox'))

    # triggers the resizing
    self.resize_apply.clicked.connect(lambda: interpolators.resize_image(self))
    # undo resizing
    self.resize_reset.clicked.connect(lambda: interpolators.reset_image(self))

    print_debug("Connectors Initialized")


def about_us(self):
    QMessageBox.about(
        self, ' About ', 'This is a Medical Image Toolbox \nCreated by Senior students from the faculty of Engineering, Cairo Uniersity, Systems and Biomedical Engineering department \n \n Created By: Mohamed Nasser ')
