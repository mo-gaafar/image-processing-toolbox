
import os
import numpy as np
import pydicom
from PIL import Image as PILImage
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import pydicom.encoders.pylibjpeg
import pydicom.encoders.gdcm
from abc import ABC, abstractmethod
from modules import interface
from modules.image import *

# abstract class for image importers using ABC module


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
        metadata = self.read_metadata(image, image_data)
        # initialize image object
        image_object = Image(data=image_data, metadata=metadata, path=path)
        return copy(image_object)

    def read_metadata(self, pil_image, image_arr) -> dict:
        # width and height data
        metadata = {}

        # if image has a third dimension
        if len(np.shape(image_arr)) == 3:
            depth = int(
                np.ceil(np.shape(image_arr)[2] * np.log2(image_arr.max()+1)))
        else:
            depth = int(np.ceil(np.log2(image_arr.max()+1)))

        metadata['Width'] = pil_image.size[0]
        metadata['Height'] = pil_image.size[1]

        # image total size
        metadata['Size'] = str(pil_image.size[0] *
                               pil_image.size[1] * depth) + ' bits'

        # bit depth data (lossless)
        metadata['Bit Depth'] = str(depth) + ' bits'

        # color mode data
        metadata['Image Color'] = pil_image.mode


        return metadata


class JPGImporter(ImageImporter):
    def import_image(self, path) -> Image:
        '''converts imported jpg path to a
        numpy array and then parses metadata
        then returns an image object'''
        # read the image
        pil_image = PILImage.open(path)
        # convert to numpy array
        image_data = np.array(pil_image)
        # parse metadata into dictionary
        metadata = self.read_metadata(pil_image, path, image_data)
        # initialize image object
        image_object = Image(data=image_data, metadata=metadata, path=path)
        return copy(image_object)

    def read_metadata(self, pil_image, path, image_arr) -> dict:
        # width and height data
        metadata = {}
        # image_arr = np.array(pil_image)

        # if image has a third dimension
        if len(np.shape(image_arr)) == 3:
            depth = int(
                np.ceil(np.shape(image_arr)[2] * np.log2(image_arr.max()+1)))
        else:
            depth = int(np.ceil(np.log2(image_arr.max()+1)))

        metadata['Width'] = pil_image.size[0]
        metadata['Height'] = pil_image.size[1]

        # image size information
        metadata['Image Size'] = str(int(
            pil_image.size[0] * pil_image.size[1] * depth)) + ' bits'
        metadata['File Size'] = str(os.path.getsize(path)*8) + ' bits'
        metadata['Compression ratio'] = str(int(
            (pil_image.size[0] * pil_image.size[1] * depth) / (os.path.getsize(path)*8) * 100)) + '%'

        # bit depth data by counting no of channels and multiplying by color depth
        metadata['Bit Depth'] = str(depth) + ' bits'

        # color mode data
        metadata['Image Color'] = pil_image.mode
        return metadata


class DICOMImporter(ImageImporter):
    def import_image(self, path) -> Image:
        '''converts imported dicom path to a
        numpy array and then parses metadata
        then returns an image object'''
        # read the image
        ds = pydicom.dcmread(path, force=True)
        # convert to numpy array
        # or whatever is the correct transfer syntax for the file
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        image_data = ds.pixel_array

        # map each element to be between 0 and 255
        # TODO: add support for more formats
        image_data = np.interp(
            image_data, (image_data.min(), image_data.max()), (0, 255))

        # parse dicom metadata into dictionary
        metadata = self.read_metadata(ds, path)
        # initialize image object
        image_object = Image(data=image_data, metadata=metadata, path=path)
        # return image object
        return copy(image_object)

    def read_metadata(self, ds, path) -> dict:
        metadata = {}
        # image width and height
        metadata['Width'] = ds.Columns
        metadata['Height'] = ds.Rows

        # image size info
        metadata['Size'] = str((os.stat(path).st_size)*8) + " bits"
        metadata['Color Depth'] = str(ds.BitsStored) + " bits"

        # dicom header data
        metadata['Image Color'] = ds.get('PhotometricInterpretation', 'N/A')
        metadata['Modality'] = ds.get('Modality', 'N/A')
        metadata['Patient Name'] = ds.get('PatientName', 'N/A')
        metadata['Patient ID'] = ds.get('PatientID, N/A')
        metadata['Body Part Examined'] = ds.get('StudyDescription, N/A')
        return metadata
