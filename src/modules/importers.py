
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
        '''converts imported bitmap path to a 
        numpy array and then parses metadata
        then returns an image object'''
        # read the image
        image = PILImage.open(path)
        # convert to numpy array
        image_data = np.array(image)
        # parse metadata into dictionary
        metadata = self.read_metadata(image, path)
        # initialize image object
        image_object = Image(data=image_data, metadata=metadata, path=path)
        return copy(image_object)

    def read_metadata(self, pil_image, path)-> dict:
        #width and height data
        metadata = {}
        metadata['Width'] = pil_image.size[0]
        metadata['Height'] = pil_image.size[1]
        #image total size
        metadata['Size'] = str(os.path.getsize(path)*8) + ' bits' 
        #bit depth data
        metadata['Bit Depth'] = str(self.get_bit_depth(np.array(pil_image))) + ' bits'
        #color mode data
        metadata['Image Color'] = pil_image.mode

        return metadata

    def get_bit_depth(self, image_data):
        return image_data.dtype.itemsize * 8
        

        
class JPGImporter(ImageImporter):
    def import_image(self, path) -> Image:
        '''converts imported jpg path to a 
        numpy array and then parses metadata
        then returns an image object'''

        # read the image
        image = PILImage.open(path)
        # convert to numpy array
        image_data = np.array(image)
        # parse metadata into dictionary
        metadata = self.read_metadata(path)
        # initialize image object
        image_object = Image(data=image_data, metadata=metadata, path=path)
        return copy(image_object)

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
        metadata['Image Color'] = PILImage.open(path).mode

        return metadata
        

class DICOMImporter(ImageImporter):
    def import_image(self, path) -> Image:
        '''converts imported dicom path to a 
        numpy array and then parses metadata
        then returns an image object'''
        # read the image
        ds = pydicom.dcmread(path, force=True)
        # convert to numpy array
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian  # or whatever is the correct transfer syntax for the file
        image_data = ds.pixel_array

        #map each element to be between 0 and 255
        #TODO: add support for more formats
        image_data = np.interp(image_data, (image_data.min(), image_data.max()), (0, 255))

        # parse dicom metadata into dictionary
        metadata = self.read_metadata(ds,path)
        # initialize image object
        image_object = Image(data=image_data, metadata=metadata, path=path)
        # return image object
        return copy(image_object)
    
    def read_metadata(self, ds,path) -> dict:
        metadata = {} 
        #image width and height
        metadata['Width'] = ds.Columns
        metadata['Height'] = ds.Rows

        #image size info
        metadata['Size'] = str((os.stat(path).st_size)*8) + " bits"
        metadata['Color Depth'] = str(ds.BitsStored) + " bits"

        #dicom header data
        metadata['Image Color'] = ds.get('PhotometricInterpretation', 'N/A')
        metadata['Modality'] = ds.get('Modality', 'N/A')
        metadata['Patient Name'] = ds.get('PatientName', 'N/A')
        metadata['Patient ID'] = ds.get('PatientID, N/A')
        metadata['Body Part Examined'] = ds.get('StudyDescription, N/A')
            
            
        return metadata

# def open_file(self, path):

#     im = PILImage.open(path)
#     im = remove_transparency(im)
#     data = np.array(im)
#     return data



