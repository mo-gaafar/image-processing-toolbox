from PyQt5.QtWidgets import QFileDialog
import numpy as np
from modules import interface
from modules.image import *
from modules.importers import *


def open_new(self, image_idx=1):
    self.filename = QFileDialog.getOpenFileName(
        None, 'open the image file', './', filter="Raw Data(*.bmp *.jpg *.dcm *.jpeg)")
    path = self.filename[0]
    print_debug("Selected path: " + path)

    if path == '':
        # raise Warning("No file selected")
        return

    # select an image importer class based on the file extension
    importer = read_importer(path)

    try:
        # import the image into an image object
        self.image1 = importer.import_image(path)

        # update the image and textbox in the viewer
        interface.display_pixmap(self, image=self.image1, window_index=0)
        interface.display_metatable(self, self.image1.get_metadata())
    except:
        QMessageBox.critical(
            self, 'Error', 'Failed to import image: make sure that the correct file format is used', QMessageBox.Ok)
        return


def read_importer(path) -> ImageImporter:
    # parse file extension
    extension = path.split('.')[-1]
    # array of supported extensions
    importers = {
        'bmp': BMPImporter(),
        'jpeg': JPGImporter(),
        'jpg': JPGImporter(),
        'dcm': DICOMImporter()
    }
    if extension in importers:
        return importers[extension]
    else:
        raise Warning("Unsupported file type")
