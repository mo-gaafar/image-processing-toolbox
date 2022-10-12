
import sys
import numpy as np
from PyQt5 import QtGui, QtWidgets, uic, QtCore
from PyQt5.QtWidgets import QTabWidget
from modules import resources, interface
from modules.utility import print_debug, print_log


import ctypes
myappid = 'mycompany.myproduct.subproduct.version' # arbitrary string
# tells windows to use the string as the app id
# so that taskbar grouping works correctly
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)


class MainWindow(QtWidgets.QMainWindow):
    ''' This is the PyQt5 GUI Main Window'''

    def __init__(self, *args, **kwargs):
        ''' Main window constructor'''

        super(MainWindow, self).__init__(*args, **kwargs)
        uic.loadUi('./resources/MainWindow.ui', self)

        # set the title and icon
        self.setWindowIcon(QtGui.QIcon('./resources/icons/icon.png'))
        self.setWindowTitle("Medical Image Viewer")

        # initialize global variables
        self.image1 = None

        # initialize ui connectors
        interface.init_connectors(self)

def main():

    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
