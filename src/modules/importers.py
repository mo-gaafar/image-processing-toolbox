
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import matplotlib.pyplot as plt
import numpy as np
from modules import interface
from PIL import Image as PILImage
from abc import ABC, abstractmethod
import pydicom
import os

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
        #width and height data
        metadata = {}
        metadata['Width'] = PILImage.open(path).size[0]
        metadata['Height'] = PILImage.open(path).size[1]
        #image total size
        metadata['Size'] = str(os.path.getsize(path)*8) + ' bits' 
        #bit depth data
        metadata['Bit Depth'] = str(PILImage.open(path).bits) + ' bits'
        #color mode data
        metadata['Color Mode'] = PILImage.open(path).mode



        return metadata
        

class DICOMImporter(ImageImporter):
    def import_image(self, path) -> Image:
        # read the image
        ds = pydicom.dcmread(path, force=True)
        # convert to numpy array
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian  # or whatever is the correct transfer syntax for the file
        data = ds.pixel_array
        # parse dicom metadata into dictionary
        metadata = self.read_metadata(ds,path)
        # initialize image object
        image_object = Image(data=data, metadata=metadata, path=path)
        # return image object
        return copy(image_object)
    
    def read_metadata(self, ds,path) -> dict:
        metadata = {} 
        #image width and height
        metadata['Width'] = ds.Columns
        metadata['Height'] = ds.Rows

        #image total size in bits
        metadata['Size'] = str((os.stat(path).st_size)*8) + " bits"

        metadata['Color Depth'] = str(ds.BitsStored) + " bits"
        metadata['Image Color'] = ds.get('PhotometricInterpretation', 'N/A')

        #dicom header data
        metadata['Modality'] = ds.get('Modality', 'N/A')
        metadata['Patient Name'] = ds.get('PatientName', 'N/A')
        metadata['Patient ID'] = ds.get('PatientID, N/A')
        metadata['Body Part Examined'] = ds.get('BodyPartExamined, N/A')
            
            
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
