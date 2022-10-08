
from PyQt5.QtWidgets import QFileDialog
import matplotlib.pyplot as plt
import numpy as np
from modules import interface
from PIL import Image as PILImage
from abc import ABC, abstractmethod
import pydicom

# from modules.utility import print_debug

from modules.image import *

#abstract class for image importers using ABC

class ImageImporter(ABC):
    '''purpose: converts imported path to a 
    numpy array and then parses metadata 
    then returns an image object'''

    @abstractmethod
    def import_image(self, path):
        raise NotImplementedError       

class BMPImporter(ImageImporter):
    def import_image(self, path) -> Image:
        # read the image
        image = PILImage.open(path)
        # convert to numpy array
        image_data = np.array(image)
        # parse metadata into dictionary
        metadata = self.read_metadata(path)
        # initialize image object
        image_object = Image(data=image_data, metadata=metadata, path=path)
        return image_object

    def read_metadata(self, path)-> dict:
        return {}
        
class JPGImporter(ImageImporter):
    def import_image(self, path) -> Image:

        # read the image
        image = PILImage.open(path)
        # convert to numpy array
        image_data = np.array(image)
        # parse metadata into dictionary
        metadata = self.read_metadata(path)
        # initialize image object
        image_object = Image(data=image_data, metadata=metadata, path=path)
        return image_object
    def read_metadata(self, path)-> dict:
        return {}
        

class DICOMImporter(ImageImporter):
    def import_image(self, path) -> Image:
        # read the image
        ds = pydicom.dcmread(path)
        # convert to numpy array
        data = ds.pixel_array
        # parse dicom metadata into dictionary
        metadata = self.read_metadata

        # initialize image object
        image_object = Image(data, metadata)
        # return image object
        return image_object
    
    def read_metadata(self, ds) -> dict:
        metadata = {}
        #image width and height
        metadata['Width'] = ds.Columns
        metadata['Height'] = ds.Rows

        #image total size in bits
        metadata['Size'] = ds.BitsAllocated + " bits"

        metadata['Color Depth'] = ds.BitsStored + " bits"
        metadata['Image Color'] = ds.PhotometricInterpretation

        #dicom header data
        metadata['Modality'] = ds.Modality
        metadata['Patient Name'] = ds.PatientName
        metadata['Patient ID'] = ds.PatientID
        metadata['Body Part Examined'] = ds.BodyPartExamined

        return metadata

# def open_file(self, path):

#     im = PILImage.open(path)
#     im = remove_transparency(im)
#     data = np.array(im)
#     return data


# def remove_transparency(im, bg_colour=(255, 255, 255)):

#     # Only process if image has transparency (http://stackoverflow.com/a/1963146)
#     if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):

#         # Need to convert to RGBA if LA format due to a bug in PIL (http://stackoverflow.com/a/1963146)
#         alpha = im.convert('RGBA').split()[-1]

#         # Create a new background image of our matt color.
#         # Must be RGBA because paste requires both images have the same format
#         # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
#         bg = PILImage.new("RGBA", im.size, bg_colour + (255,))
#         bg.paste(im, mask=alpha)
#         return bg

#     else:
#         return im
