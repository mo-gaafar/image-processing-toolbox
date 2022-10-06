
from turtle import onrelease
from PyQt5.QtWidgets import QTabWidget, QAction, QPushButton, QSlider, QComboBox, QLCDNumber, QMessageBox
from PyQt5.QtGui import *
from scipy.fft import fftshift
from modules import openfile
from modules import mixer
import numpy as np
from PIL import Image as PILImage
from PyQt5 import QtGui


def update_display(self, display_keys=[]):
    '''Updates multiple images in the ui'''
    for display in display_keys:
        if display == 'image1':
            display_pixmap(
                self, display, self.image1_configured.get_original_image())
        if display == 'image1_component':
            display_pixmap(
                self, display,  self.image1_configured.get_ft_plot())
        if display == 'image2':
            display_pixmap(
                self, display, self.image2_configured.get_original_image())
        if display == 'image2_component':
            display_pixmap(
                self, display, self.image2_configured.get_ft_plot())
        if display == 'output1':
            display_pixmap(self, display, self.mixer.get_mixed_image_data())
        if display == 'output2':
            display_pixmap(self, display, self.mixer.get_mixed_image_data())


def display_pixmap(self, display_key: str = 'image1', image_data: np.array = []):
    '''
    Display the image data in a QPixmap given a key to the display_reference_dict
    '''
    if image_data.size == 0:
        raise Warning("No image data to display")
        return
    if display_key in self.display_reference_dict:

        # then convert it to image format
        data = PILImage.fromarray(image_data.astype(np.uint8))
        # save the image file as png
        data.save('temp.png')
        # display saved image in Qpixmap

        self.display_reference_dict[display_key].setPixmap(
            QtGui.QPixmap("temp.png"))
        self.display_reference_dict[display_key].show()
    else:
        raise Warning("Invalid display key")
        return


def init_connectors(self):
    '''Initializes all event connectors and triggers'''

    # ''' Menu Bar'''
    # self.actionOpen = self.findChild(QAction, "actionOpen")
    # self.actionOpen.triggered.connect(
    #     lambda: openfile.browse_window(self))

    # self.actionExport = self.findChild(QAction, "actionExport")
    # self.actionExport.triggered.connect(
    #     lambda: openfile.export_summed_signal(self))

    # self.actionAbout_Us = self.findChild(QAction, "actionAbout_Us")
    # self.actionAbout_Us.triggered.connect(
    #     lambda: about_us(self))

    # self.WindowTabs = self.findChild(QTabWidget, "WindowTabs")

    ''' Browse buttons'''
    # TODO dont forget to add new argument to browse_window

    # the index argument maps each function to its respective slot
    #
    self.insert_image1_pushButton.clicked.connect(
        lambda: openfile.browse_window(self, 1))
    self.insert_image2_pushButton.clicked.connect(
        lambda: openfile.browse_window(self, 2))

    ''' Image Preview Component Dropdowns'''
    # 1. on index change
    # 2. change component in image configuration
    # 3. display image configuration component using pixmap?

    self.image1_component_comboBox.currentIndexChanged.connect(
        lambda: self.image1_configured.set_selected_feature(
            feature=self.image1_component_comboBox.currentText())
    )
    self.image2_component_comboBox.currentIndexChanged.connect(
        lambda: self.image2_configured.set_selected_feature(
            feature=self.image2_component_comboBox.currentText())
    )

    self.image1_component_comboBox.currentIndexChanged.connect(
        lambda: update_display(self, display_keys=['image1_component']))

    self.image2_component_comboBox.currentIndexChanged.connect(
        lambda: update_display(self, display_keys=['image2_component']))

    ''' Image Mixer'''
    # uses image configurations already made
    # selects index of images to be mixed
    # TODO how to deal with image 1, image 1 for example, in the most efficient way?

    # TODO Add output selection functionality somewhere
    # outside class function then refresh selected display
    self.mixer_output_comboBox.currentIndexChanged.connect(
        lambda: mixer.update_mixer(self)
    )

    # inside class function then refresh display (from class info)
    self.mixer_component1_comboBox.currentIndexChanged.connect(
        lambda: mixer.update_mixer(self)
    )
    self.mixer_component2_comboBox.currentIndexChanged.connect(
        lambda: mixer.update_mixer(self)
    )

    self.mixed_component1_comboBox.currentIndexChanged.connect(
        lambda: mixer.update_mixer(self)
    )
    self.mixed_component2_comboBox.currentIndexChanged.connect(
        lambda: mixer.update_mixer(self)
    )

    self.mixer_component1_horizontalSlider.sliderReleased.connect(
        lambda: mixer.update_mixer(self)
    )
    self.mixer_component2_horizontalSlider.sliderReleased.connect(
        lambda: mixer.update_mixer(self)
    )


def about_us(self):
    QMessageBox.about(
        self, ' About ', 'This is a nyquist theory illustrator \nCreated by junior students from the faculty of Engineering, Cairo Uniersity, Systems and Biomedical Engineering department \n \nTeam members: \n-Mohammed Nasser \n-Abdullah Saeed \n-Zeyad Mansour \n-Mariam Khaled \n \nhttps://github.com/mo-gaafar/Nyquist_Theory_Illustrator.git ')
