"""
deal with the origin
convert medicine image to numpy
nii, dcm, mhd
"""
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from glob import glob
import os
import pydicom
from utils.visualization import show_single_view
from utils.message import Msg


class Convert:
    def __init__(self):
        pass

    '''base image_root select convert'''
    def select(self, image_root: str):
        # dicm file dir
        if os.path.isdir(image_root):
            images_np = self.dcm_2_np(image_root)
        else:
            if image_root.endswith("nii") or image_root.endswith("gz"):
                images_np = self.nii_2_np(image_root)
            elif image_root.endswith("mhd"):
                images_np = self.mhd_2_np(image_root)
            else:
                pass
        # print(images_np.max(),images_np.min())
        return images_np

    @staticmethod
    def nii_2_np(image_root: str):
        """
        nii type convert to numpy
        :param image_root:
        :return: image shape [s,h,w]
        """
        nib_image = nib.load(image_root)
        # print(nib_image.header)
        # 返回spacing
        space = nib_image.header["pixdim"][1:4]
        image_data = nib_image.get_data()
        images_numpy = np.array(image_data)
        # image = sitk.ReadImage(image_root)
        # images_numpy = sitk.GetArrayFromImage(image).astype(np.float32)
        # origin = image.GetOrigin()
        # direction = image.GetDirection()
        # space = np.array(image.GetSpacing())
        # cases_dic = {'img_arr': images_numpy, 'origin': origin, 'direction': direction, 'space': space}

        return images_numpy, space

    @staticmethod
    def dcm_2_np(image_root: str):
        """ fix xcd
        dicom type convert to numpy
        :param image_root:
        :return:image shape [s,h,w]
        """
        images_numpy = [pydicom.read_file(image_root + filename)
                  for filename in os.listdir(image_root) if filename.endswith(".dcm")]
        # Sort the dicom slices in their respective order
        images_numpy.sort(key=lambda x: int(x.InstanceNumber),reverse=True)
        # print(images_numpy[1].RescaleIntercept, images_numpy[1].RescaleSlope)
        # Get the pixel values for all the slices
        images_numpy = np.stack([s.pixel_array for s in images_numpy])

        return images_numpy

    @staticmethod
    def mhd_2_np(image_root):
        """
        mhd type convert to numpy
        :param image_root:
        :return:image shape [s,h,w]
        """
        image = glob(image_root)
        image_mhd = sitk.ReadImage(image)
        images_numpy = sitk.GetArrayFromImage(image_mhd).squeeze()
        # print(images_numpy.max(), images_numpy.min())
        return images_numpy


convert = Convert()

if __name__ == "__main__":
    msg = Msg()


