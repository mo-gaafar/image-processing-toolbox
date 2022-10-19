from PyQt5.QtWidgets import QFileDialog
import numpy as np
from modules import interface
from modules.image import *
from modules.importers import *



def browse_window(self, image_idx=1):
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
    except :
        QMessageBox.critical(
            self, 'Error', 'Failed to import image: make sure that the correct file format is used', QMessageBox.Ok)
        return
    # update the image and textbox in the viewer
    interface.refresh_display(self)

def read_importer(path) -> ImageImporter:
    #parse file extension
    extension = path.split('.')[-1]
    #array of supported extensions
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
