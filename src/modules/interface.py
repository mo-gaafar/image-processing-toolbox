
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
        dimensions_string = str(data_arr.shape[0]) +"X"+ str(data_arr.shape[1])+ " px"
        self.resize_original_dim_textbox.setText(dimensions_string)
    elif selector == 'resized':
        dimensions_string = str(data_arr.shape[0]) +"X"+ str(data_arr.shape[1])+ " px"
        self.resize_resized_dim_textbox.setText(dimensions_string)

        


def get_user_input(self):
    '''Gets the user input from the GUI and returns it as a dictionary'''
    user_input = {}
    user_input['resize factor'] = self.resize_spinbox.value()
    user_input['interpolation method'] = self.interpolate_combobox.currentText()
    return user_input


def refresh_display(self):
    ''' Updates the user interface with the current image and data'''
    try:
        display_pixmap(self, image=self.image1)
    except TypeError:
        # error message pyqt
        QMessageBox.critical(
            self, 'Error', 'Error Loading Image: Unsupported Image Format')

    display_metatable(self)


def print_statusbar(self, message):
    self.statusbar.showMessage(message)


def display_pixmap(self, image):
    '''Displays the image data in the image display area'''
    # then convert it to image format
    image_data = image.get_pixels()

    print_debug("Displaying Image")
    print_debug(image_data.shape)

    # show image using matplotlib
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(image_data, cmap='gray')
    plt.show()

    qim = None

    if len(image_data.shape) == 2:
        im = PILImage.fromarray(image_data)
        if im.mode != 'L':
            im = im.convert('L')

        qim = im.toqimage()

        # # get the shape of the array
        # nframes, height, width = np.shape(self.image_data)

        # # calculate the total number of bytes in the frame
        # totalBytes = self.image_data[0].nbytes

        # # divide by the number of rows
        # bytesPerLine = int(totalBytes/height)

        # # create a QImage object
        # qim = QtGui.QImage(self.image_data[0], width, height,
        #                    bytesPerLine, QImage.Format_Grayscale8)

    elif len(image_data.shape) == 3:
        im = PILImage.fromarray(image_data)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        data = im.tobytes()
        qim = QtGui.QImage(
            data, im.size[0], im.size[1], QtGui.QImage.Format_RGB888)

    # convert the image to binary in RGB format

    # display saved image in Qpixmap
    self.image1_widget.setPixmap(QtGui.QPixmap.fromImage(qim))
    self.image1_widget.show()


def display_metatable(self):
    '''Displays the metadata in QTableWidget'''
    f_metadata = self.image1.get_metadata()
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
    self.actionOpen.triggered.connect(lambda: openfile.open_file(self))
    self.actionSave_As.triggered.connect(lambda: openfile.save_file(self))
    self.actionSave.triggered.connect(lambda: openfile.save_file(self))

    # View Menu
    self.actionMetadata.triggered.connect(lambda: self.toggle_metadata_tab())
    self.actionResizer.triggered.connect(lambda: self.toggle_resize_tab())

    # Help Menu
    self.actionAbout.triggered.connect(lambda: about_us(self))

    # self.actionAbout_Us = self.findChild(QAction, "actionAbout_Us")
    # self.actionAbout_Us.triggered.connect(
    #     lambda: about_us(self))

    # self.WindowTabs = self.findChild(QTabWidget, "WindowTabs")

    ''' browse buttons'''
    # the index argument maps each function to its respective slot
    self.insert_image1_pushButton.clicked.connect(
        lambda: openfile.browse_window(self, 1))

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
