import SimpleITK as sitk
import numpy as np
import os
from utils.message import Msg
import time

class Save:
    def __init__(self):
        pass

    def save_2_nii(self, images: list, save_path: str, patient_name):
        """待定"""
        msg = Msg()
        local_time = time.localtime()
        format_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
        filename = patient_name+format_time + ".nii"
        # filename = patient_name + ".nii"
        nii = os.path.join(save_path, filename)
        result = sitk.GetImageFromArray(images)
        # nii为存储路径
        sitk.WriteImage(result, nii)
        msg.norm("info","save nii")

