import SimpleITK as sitk
import numpy as np
import os
from utils.message import Msg
import time
from utils.config import config
import torch


class Save:
    def __init__(self):
        pass

    @staticmethod
    def save_2_nii(images, save_path: str, patient_name="kidney"):
        """待定"""
        msg = Msg()
        local_time = time.localtime()
        format_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
        filename = patient_name+format_time + ".nii"
        # filename = patient_name + ".nii"
        try:
            os.mkdir(save_path)
        except OSError:
            pass
        if type(images) is not np.ndarray:
            images = images.cpu().numpy()
        if len(images.shape) == 4:
            images = images.squeeze(axis=0)
        print(images.shape)
        nii = os.path.join(save_path, filename)
        result = sitk.GetImageFromArray(images)
        # nii为存储路径
        sitk.WriteImage(result, nii)
        msg.norm("info", f"save {patient_name}")


if __name__ == "__main__":
    image_path = os.path.join(config.crop_origin_image, "case_00049-002-03.pth")
    image_tensor = torch.load(image_path)
    print(image_tensor.shape)
    Save.save_2_nii(image_tensor, config.save_path)