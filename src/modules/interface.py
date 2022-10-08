
from turtle import onrelease
from PyQt5.QtWidgets import QTabWidget, QAction, QPushButton, QSlider, QComboBox, QLCDNumber, QMessageBox
from PyQt5.QtGui import *
import numpy as np
from PIL import Image as PILImage
from PyQt5 import QtGui
import os
from modules.utility import print_debug
from modules import openfile



def refresh_display(self):
    ''' Updates the user interface with the current image and data'''
    display_pixmap(self, image_data=self.image1.get_pixels())
    display_list(self)

def display_pixmap(self, image_data):
    '''Displays the image data in the image display area'''
    # then convert it to image format
    data = PILImage.fromarray(image_data.astype(np.uint8))
    # save the image file as png
    data.save('temp.png')
    # display saved image in Qpixmap
    self.image1_widget.setPixmap(QtGui.QPixmap("temp.png"))
    self.image1_widget.show()
    # delete the temporary image file
    os.remove('temp.png')

#TODO: make it display it in a qlistwidget
def display_list(self):
    '''Displays the metadata in qlabel'''
    f_metadata = self.image1.get_formatted_metadata()
    print("metadata: ", f_metadata)
    self.metadata_widget.setText(f_metadata)


def init_connectors(self):
    '''Initializes all event connectors and triggers'''

    # ''' Menu Bar'''

    # self.actionAbout_Us = self.findChild(QAction, "actionAbout_Us")
    # self.actionAbout_Us.triggered.connect(
    #     lambda: about_us(self))

    # self.WindowTabs = self.findChild(QTabWidget, "WindowTabs")

    ''' browse buttons'''
    # the index argument maps each function to its respective slot
    self.insert_image1_pushButton.clicked.connect(
        lambda: openfile.browse_window(self, 1))

    print_debug("Connectors Initialized")

def about_us(self):
    QMessageBox.about(
        self, ' About ', 'This is a Medical Image Viewer \nCreated by Senior students from the faculty of Engineering, Cairo Uniersity, Systems and Biomedical Engineering department \n \n Created By: Mohamed Nasser ')

