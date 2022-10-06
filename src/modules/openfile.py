# OLD CODE... REMOVE THIS COMMENT WHEN DONE MODIFYING

from PyQt5.QtWidgets import QFileDialog
import matplotlib.pyplot as plt
import numpy as np
from modules import interface
from PIL import Image as PILImage

# from modules.utility import print_debug

from modules.mixer import *


def browse_window(self, image_idx=1):
    self.filename = QFileDialog.getOpenFileName(
        None, 'open the signal file', './', filter="Raw Data(*.bmp *.jpg *.png)")
    path = self.filename[0]
    print_debug("Selected path: " + path)

    if path == '':
        raise Warning("No file selected")
        return

    data = open_file(self, path)

    if (image_idx == 1):
        self.image1_configured = ImageConfiguration(
            Image(path=path, data=data, fourier_enable=True))

        # initialize feature selection
        feature_idx = self.image1_component_comboBox.currentIndex()
        self.image1_configured.set_selected_feature(index=feature_idx)

        # TODO for testing purposes abstract later in interface, update display
        interface.update_display(
            self, display_keys=['image1', 'image1_component'])

    elif (image_idx == 2):

        self.image2_configured = ImageConfiguration(
            Image(path=path, data=data, fourier_enable=True))

        # initialize feature selection
        feature_idx = self.image2_component_comboBox.currentIndex()
        self.image2_configured.set_selected_feature(index=feature_idx)

        # TODO for testing purposes abstract later in interface, update display
        interface.update_display(
            self, display_keys=['image2', 'image2_component'])


def open_file(self, path):

    im = PILImage.open(path)
    im = remove_transparency(im)
    data = np.array(im)
    return data


def remove_transparency(im, bg_colour=(255, 255, 255)):

    # Only process if image has transparency (http://stackoverflow.com/a/1963146)
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):

        # Need to convert to RGBA if LA format due to a bug in PIL (http://stackoverflow.com/a/1963146)
        alpha = im.convert('RGBA').split()[-1]

        # Create a new background image of our matt color.
        # Must be RGBA because paste requires both images have the same format
        # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
        bg = PILImage.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg

    else:
        return im
