
from turtle import onrelease
from typing import Type
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QAction, QPushButton, QMessageBox
from PyQt5.QtGui import *
import numpy as np
from PIL import Image as PILImage
from PyQt5 import QtGui
import os
from modules.utility import print_debug
from modules import openfile



def refresh_display(self):
    ''' Updates the user interface with the current image and data'''
    try:
        display_pixmap(self, image=self.image1)
    except TypeError:
        # error message pyqt
        QMessageBox.critical(self,'Error', 'Error Loading Image: Unsupported Image Format')

    display_list(self)


def display_pixmap(self, image):
    '''Displays the image data in the image display area'''
    # then convert it to image format
    image_data = image.get_pixels()
    
    im = PILImage.fromarray(image_data)

    # convert the image to binary in RGB format

    if im.mode != 'RGB':
        im = im.convert('RGB')

    data = im.tobytes()
    qim = QtGui.QImage(data, im.size[0], im.size[1], QtGui.QImage.Format_RGB888)

    # display saved image in Qpixmap
    self.image1_widget.setPixmap(QtGui.QPixmap.fromImage(qim))
    self.image1_widget.show()

#TODO: make it display in a QTableWidget
def display_list(self):
    '''Displays the metadata in QTableWidget'''
    f_metadata = self.image1.get_metadata()
    # f_metadata = self.image1.get_formatted_metadata()
    self.metadata_tablewidget.clear()
    self.metadata_tablewidget.setRowCount(len(f_metadata))
    self.metadata_tablewidget.setColumnCount(2)
    self.metadata_tablewidget.setHorizontalHeaderLabels(['Property', 'Value'])
    for i, (key, value) in enumerate(f_metadata.items()):
        self.metadata_tablewidget.setItem(i, 0, QTableWidgetItem(str(key)))
        self.metadata_tablewidget.setItem(i, 1, QTableWidgetItem(str(value)))
    # self.metadata_widget.setText(f_metadata)


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

