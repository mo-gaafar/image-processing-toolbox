
from PyQt5 import QtGui, QtWidgets, uic, QtCore
from PyQt5.QtWidgets import QTabWidget
from modules import resources, interface
import numpy as np
from modules.utility import print_debug, print_log
import sys
from modules.viewer import ImageConfiguration, ImageMixer, Image, ImageFFT


#TODO: plan out the structure of the application
#TODO: plan out the user interface
#TODO: implement the user interface

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

        self.image = Image(np.zeros((1, 1, 1)))



        print_debug("Connectors Initialized")


def main():

    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
