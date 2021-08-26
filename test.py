"""
1. test model accurate
"""
from utils.visualization import show_single_view,show_views
import numpy as np
import gc
import torch
from utils.config import config, Config
import os
import SimpleITK as sitk
from utils.data_prepare_3d import DataPrepare3D
from utils.config import CoarseConfig
import pandas as pd
import json
from utils.data_utils import save_nii, z_resample
from tqdm import tqdm
from utils.data_prepare_3d import DataPrepare3D
import math
from utils.data_utils import data_windowing
from utils.config import CoarseConfig

def read_npy():
    image = np.load(f"{config.val_mask_crop}/0_0_case_00288_.npy")
    show_single_view(image, cmap="gray")
    image = np.load(f"{config.val_image_crop}/0_0_case_00288_.npy")
    show_single_view(image, cmap="gray")


def get_memory(i):
    del i
    gc.collect()


def crop_slice():
    config = CoarseConfig()
    data_prepare = DataPrepare3D(config)
    cases_kidneys_info = dict()
    cubes_shape = dict()
    with open(data_prepare.cases_spacing_json) as f:
        cases_spacing = json.load(f)
    with tqdm(total=len(cases_spacing), desc=f"crop cube", unit="case") as pbar:
        for i, case_name in enumerate(cases_spacing):
            mask_path = os.path.join(data_prepare.origin_mask, case_name + ".nii.gz")
            image_path = os.path.join(data_prepare.origin_image, case_name + ".nii.gz")

            # read origin case data
            # 512
            mask_nii = sitk.ReadImage(mask_path)
            image_nii = sitk.ReadImage(image_path)
            mask_arr = sitk.GetArrayFromImage(mask_nii).astype(np.uint8).transpose((2, 1, 0))
            image_arr = sitk.GetArrayFromImage(image_nii).astype(np.float).transpose((2, 1, 0))

            image_arr, new_spacing = z_resample(image_arr, cases_spacing[case_name], re_thickness=data_prepare.thickness)

            mask_arr, _ = z_resample(mask_arr, cases_spacing[case_name], re_thickness=data_prepare.thickness,)
            # print(mask_arr.shape)
            kidney_area = data_prepare.get_kidney_area(mask_arr)
            # print(kidney_area)
            # case_kidneys_info = {"kidney":kidney_area}
            case_kidneys_info = dict()

            case_kidneys_info["old_spacing"] = cases_spacing[case_name]
            print(image_arr.shape, cases_spacing[case_name])
            for j in range(len(kidney_area)):

                mask_crop, img_crop, is_normal, crop_edge = data_prepare.crop_area(image_arr, mask_arr,
                                                                                   kidney_area[j])
                # case_kidneys_info["crop_edge"] = crop_edge
                img_crop, new_spacing = z_resample(img_crop, cases_spacing[case_name],
                                                   re_thickness=data_prepare.thickness, re_xy=data_prepare.resize_xy)
                mask_crop, _ = z_resample(mask_crop, cases_spacing[case_name],
                                          re_thickness=data_prepare.thickness, re_xy=data_prepare.resize_xy)
                kidney_name = f"{case_name}_kidney{j}"
                # print(mask_crop.shape)
                save_nii(config.crop_cube_mask, mask_crop, cases_spacing[case_name],
                         f"{kidney_name}.nii.gz")
                save_nii(config.crop_cube_image, img_crop, cases_spacing[case_name],
                         f"{kidney_name}.nii.gz")
                kidney_area[j]["crop_edge"] = crop_edge
                cubes_shape[kidney_name] = mask_crop.shape
            pbar.update()
            case_kidneys_info["new_spacing"] = new_spacing
            case_kidneys_info["kidney"] = kidney_area
            cases_kidneys_info[case_name] = case_kidneys_info

    data_prepare.info_to_json(config.kidney_info_json, cases_kidneys_info)
    data_prepare.info_to_json(config.cube_shape_json, cubes_shape)
    print("end")


def crop_fine_cube(cube_name):
    """

    :param cases_name(dict):{"cases_name":["","",....]}
    :return:
    """
    coarse_config = CoarseConfig()
    img_output = coarse_config.crop_64cube_image
    mask_output = coarse_config.crop_64cube_mask
    cube_root = coarse_config.crop_64cube_path
    if not os.path.exists(cube_root):
        os.mkdir(cube_root)
    if not os.path.exists(img_output):
        os.mkdir(img_output)
    if not os.path.exists(mask_output):
        os.mkdir(mask_output)
    # print(len(cube_name["cases_name"]))
    for case_name in cube_name["cases_name"]:
        img_path = os.path.join(coarse_config.crop_cube_image, case_name + ".nii.gz")
        mask_path = os.path.join(coarse_config.crop_cube_mask, case_name + ".nii.gz")
        # read origin case data
        # 512
        img_nii = sitk.ReadImage(img_path)
        mask_nii = sitk.ReadImage(mask_path)
        img_arr = sitk.GetArrayFromImage(img_nii).astype(np.float32)
        mask_arr = sitk.GetArrayFromImage(mask_nii).astype(np.uint8)
        # remove the diff shape ,h and w
        # print(img_arr.shape, ct_info[case_name])
        # Windowing

        img_arr = data_windowing(img_arr, coarse_config.windowing)
        # mask_arr = data_windowing(mask_arr, coarse_config.windowing)
        # z pading
        if img_arr.shape[0] < coarse_config.layer_thick:
            p = coarse_config.layer_thick - img_arr.shape[0]
            img_arr = np.pad(img_arr, ((0, p), (0, 0), (0, 0)), 'minimum')
            mask_arr = np.pad(mask_arr, ((0, p), (0, 0), (0, 0)), 'minimum')
        print(case_name, img_arr.shape, mask_arr.shape)
        if img_arr.shape[1] != coarse_config.area_size[1] or img_arr.shape[2] != coarse_config.area_size[
            2] or img_arr.shape != mask_arr.shape:
            continue
        # print(img_arr.shape)
        # crop data block
        voxel_z = mask_arr.shape[0]
        target_gen_size = coarse_config.layer_thick
        range_val = int(math.ceil((voxel_z - target_gen_size) / coarse_config.stride) + 1)

        for i in range(range_val):
            fileid = case_name + "-" + '%03d' % i
            print(fileid)
            start_num = i * coarse_config.stride
            end_num = start_num + target_gen_size
            print('total:', voxel_z, 'start_end:', start_num, end_num)
            if end_num <= voxel_z:
                # 数据块长度没有超出x轴的范围,正常取块
                img = img_arr[start_num:end_num, :, :]
                mask = mask_arr[start_num:end_num, :, :]
            else:
                # 数据块长度超出x轴的范围, 从最后一层往前取一个 batch_gen_size 大小的块作为本次获取的数据块
                img = img_arr[(voxel_z - target_gen_size):voxel_z, :, :]
                mask = mask_arr[(voxel_z - target_gen_size):voxel_z, :, :]

            img = torch.from_numpy(img)
            mask = torch.from_numpy(mask.astype(np.uint8))

            img = torch.unsqueeze(img, dim=0)
            mask = torch.unsqueeze(mask, dim=0)

            torch.save(img, os.path.join(img_output, fileid + '.pth'))
            torch.save(mask, os.path.join(mask_output, fileid + '.pth'))
            print(os.path.join(mask_output, fileid + '.pth'))

            assert img.shape[1] == coarse_config.layer_thick and mask.shape[1] == coarse_config.layer_thick
            if len(coarse_config.trans_types) > 0 and mask.any() > 0:
                # augmentation
                for j, trans in enumerate(coarse_config.trans_types):
                    seed = np.random.randint(2147483647)
                    img_aug, mask_aug = DataPrepare3D.data_augmentation(img, mask, trans, seed)
                    # print()
                    if torch.max(mask_aug) <= torch.max(torch.unique(mask))\
                            and torch.min(mask_aug) >= torch.min(torch.unique(mask)):
                        img_aug = img
                        mask_aug = mask
                    # print(img_aug.shape, mask_aug.shape)
                    # xcd:bug???
                    assert img_aug.shape[1] == coarse_config.layer_thick and mask_aug.shape[1] == coarse_config.layer_thick, \
                        f"image:{img_aug.shape}, layer_thick_{coarse_config.layer_thick}"

                    aug_name = fileid + '-' + '%02d' % j + '.pth'
                    torch.save(img_aug, os.path.join(img_output, aug_name))
                    torch.save(mask_aug, os.path.join(mask_output, aug_name))
        print('*' * 30)
        print("end")


def crop_origin_data():
    config = Config()
    data_prepare = DataPrepare3D(config)
    cases_name = pd.read_json(config.case_name_json)
    data_prepare.transform_to_3d(cases_name, config.crop_origin_image, config.crop_origin_mask)


def deal_223():
    case_name = "case_00223.nii.gz"
    patient = "case_00223"
    data_path = "/public/datasets/kidney/kits21/kits21/data"
    image_case_path = os.path.join(data_path, patient, "imaging.nii.gz")
    mask_case_path = os.path.join(data_path, patient, "aggregated_MAJ_seg.nii.gz")
    image = os.path.join(config.image_3d_path, case_name)
    mask = os.path.join(config.mask_3d_path, case_name)
    mask_nii = sitk.ReadImage(mask_case_path)
    img_nii = sitk.ReadImage(image_case_path)
    img_arr = sitk.GetArrayFromImage(img_nii).astype(np.float32)
    mask_arr = sitk.GetArrayFromImage(mask_nii).astype(np.uint8)
    print(mask_arr.shape)
    img_arr = img_arr[:, :, 50:120]
    mask_arr = mask_arr[:, :, 50:120]
    save_nii(image, img_arr)
    save_nii(mask, mask_arr)


def crop_origin_data_thr():
    config = Config()
    data_prepare = DataPrepare3D(config)
    cases_name = pd.read_json(config.case_name_json)
    from threading import Thread
    class Metrics_thread(Thread):
        def __init__(self, cases_name, crop_origin_image, crop_origin_mask):
            Thread.__init__(self)
            self.result = 0
            self.cases_name = cases_name
            self.crop_origin_image = crop_origin_image
            self.crop_origin_mask = crop_origin_mask

        def run(self):
            # print("start")
            self.result = data_prepare.transform_to_3d(self.cases_name,  self.crop_origin_image, self.crop_origin_mask)

        def result(self):
            return self.result

    data_thread_list = list()
    for i in range(60):
        data_thread_list.append(Metrics_thread(cases_name[i*5:(i+1)*5],
                                               config.crop_origin_image, config.crop_origin_mask))
        data_thread_list[i].start()
    for i in range(60):
        data_thread_list[i].join()
        print("end")


def crop_fine_cube_thread():
    config = Config()
    data_prepare = DataPrepare3D(config)
    with open(config.cube_name_json) as f:
        cases_name = json.load(f)
        cases_name = json.loads(cases_name)
    # cases_name = pd.read_json(config.cube_name_json)

    from threading import Thread
    class Metrics_thread(Thread):
        def __init__(self, cube_name):
            Thread.__init__(self)
            self.result = 0
            self.cube_name = cube_name

        def run(self):
            # print("start")
            self.result = crop_fine_cube(self.cube_name)

        def result(self):
            return self.result

    data_thread_list = list()
    for i in range(30):
        if (i + 1) * 20 < len(cases_name["cases_name"]):
            data_thread_list.append(Metrics_thread({"cases_name":cases_name["cases_name"][i*20:(i+1)*20]}))
        else:
            data_thread_list.append(Metrics_thread({"cases_name":cases_name["cases_name"][i*20:]}))
        data_thread_list[i].start()
    for i in range(30):
        data_thread_list[i].join()
        print("end")


if __name__ == "__main__":
    # config = Config()
    # # cases_name = pd.read_json(config.cube_name_json)
    # with open(config.cube_name_json) as f:
    #     cube_name = json.load(f)
    #     cube_name = json.loads(cube_name)
    # print(cube_name)
    crop_origin_data_thr()

