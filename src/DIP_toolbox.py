""" This is the main file for the DIP toolbox. It contains the main window and the main function.
Created by: M. Nasser Gaafar, 2022 for the course Digital Image Processing (DIP) at Cairo Univesrity
"""

import sys
import numpy as np
from PyQt5 import QtGui, QtWidgets, uic, QtCore
from PyQt5.QtWidgets import QTabWidget
from modules import resources, interface
from modules.utility import print_debug, print_log

MY_APP_ID = u'cu.dip_tooblox.v0.7'  # arbitrary string

if sys.platform == 'win32':
    import ctypes

    # tells windows to use the string as the app id
    # so that taskbar grouping works correctly
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(MY_APP_ID)


if sys.platform == 'darwin':
    # workaround for the issue with the menu bar on macOS
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_DontUseNativeMenuBar)


class MainWindow(QtWidgets.QMainWindow):
    ''' This is the PyQt5 GUI Main Window'''

    def __init__(self, *args, **kwargs):
        ''' Main window constructor'''

        super(MainWindow, self).__init__(*args, **kwargs)
        uic.loadUi('./resources/MainWindow.ui', self)


        # set the title and icon
        # set app icon
        app_icon = QtGui.QIcon('./resources/icons/icon.png')
        app_icon.addFile('./resources/icons/16x16.png', QtCore.QSize(16, 16))
        app_icon.addFile('./resources/icons/24x24.png', QtCore.QSize(24, 24))
        app_icon.addFile('./resources/icons/32x32.png', QtCore.QSize(32, 32))
        app_icon.addFile('./resources/icons/48x48.png', QtCore.QSize(48, 48))
        app_icon.addFile('./resources/icons/256x256.png',
                         QtCore.QSize(256, 256))
        self.setWindowIcon(app_icon)
        self.setWindowTitle("Medical Image Toolbox")

        # initialize global variables
        self.image1 = None

        # initialize ui connectors
        interface.init_connectors(self)

    def output_click_statusbar(self, event):

        x = event.xdata
        y = event.ydata
        str = f"Point clicked at {x}, {y} on plot "
        interface.print_statusbar(self, str)


def main():

    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
