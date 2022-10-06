# https://namingconvention.org/python/ use the pythonic naming convention here (friendly reminder)

from PyQt5 import QtGui, QtWidgets, uic, QtCore
from PyQt5.QtWidgets import QTabWidget
from modules import resources, interface
import numpy as np
from modules.utility import print_debug, print_log
import sys
from modules.mixer import ImageConfiguration, ImageMixer, Image, ImageFFT


class MainWindow(QtWidgets.QMainWindow):
    ''' This is the PyQt5 GUI Main Window'''

    def __init__(self, *args, **kwargs):
        ''' Main window constructor'''

        super(MainWindow, self).__init__(*args, **kwargs)
        uic.loadUi('./resources/MainWindow.ui', self)

        # set the title and icon
        self.setWindowIcon(QtGui.QIcon('./resources/icons/icon.png'))
        self.setWindowTitle("Image Mixer")

        # initialize arrays and variables

        interface.init_connectors(self)

        self.image1_configured: ImageConfiguration = None
        self.image2_configured: ImageConfiguration = None
        self.mixer: ImageMixer = None

        # dictionary containing all image display labels in ui
        self.display_reference_dict = {
            'image1': self.image1_widget,
            'image1_component': self.image1_component_widget,
            'image2': self.image2_widget,
            'image2_component': self.image2_component_widget,
            'output1': self.output1_widget,
            'output2': self.output2_widget,
        }

        # initialize global image mixer object
        # self.image_mixer = ImageMixer()

        print_debug("Connectors Initialized")


def main():

    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
