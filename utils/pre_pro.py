import numpy as np
import cv2
import sys
from data.BasisDataset import BaseDataset
sys.path.append("../")
from utils.visualization import show_image_hist, show_single_view
import SimpleITK as sitk
from utils.config import config
import json
import os
from threading import Thread
from matplotlib import pyplot as plt
from utils.convert import Convert
from itertools import groupby
from utils.data_prepare_3d import DataPrepare3D
import torch
import pandas as pd
import matplotlib.pyplot as plt

# normalize HU to [1,255]
def normalize(image):
    ymax = 255
    ymin = 0

    image[image<-1000] = -1000
    image[image > 3000] = 3000
    image = image + 1000
    xmax = image.max()
    xmin = image.min()
    # print(xmax, xmin)
    # norm_image = (ymax - ymin) * (image - xmin) / (xmax - xmin) + ymin

    cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
    return image


# resample the CT image
def resize_image_itk(itkimage, resamplemethod=sitk.sitkNearestNeighbor):
    """
    resize image
    :param itkimage:
    :param resamplemethod:
    :return: resize image
    """
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  # 原来的体素块尺寸
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(config.new_size, float)
    factor = originSize / newSize
    # newSpacing = originSpacing * factor
    new_spacing = config.spacing
    newSize = newSize.astype(np.int)  # spacing肯定不能是整数
    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(list(new_spacing))
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    return itkimgResampled


# origin sample
def origin_sample():
    pass


# get json information
def read_json(json_file: str) -> dict:
    delineate = dict()
    with open(json_file, 'r') as load_f:
        load_dict = json.load(load_f)
        # print(load_dict['annotations'])
        # print(len(load_dict['annotations']))
        annotations = load_dict['annotations']
        for anno_id in range(len(annotations)):
            # 存在多个相同的frame
            frame = annotations[anno_id]['frame']
            spatial_payload = annotations[anno_id]['spatial_payload']
            if frame in delineate.keys():
                delineate[frame].append(spatial_payload)
            else:
                delineate[frame] = [spatial_payload]

    return delineate


# show a patient CT images spatial_payload
def show_contours(image_root, delineations: list, patient_id: int):
    """
    :param image_root: location of nii.gz file
    :param delineations: dict(){frame:point} the frame is the CT slice index
    :return: mask nii.gz  of artery
    """
    images = sitk.ReadImage(image_root)
    images_array = sitk.GetArrayFromImage(images).transpose((2, 1, 0))
    print(len(delineations))
    for delineation_index in range(len(delineations)):

        delineation = delineations[delineation_index]
        for key in delineation:
            # print(key)

            contours = delineation[key]
            # print(type(contours))
            contour = list()
            # print(key, len(contours))
            for i in range(len(contours)):
              contour.append(np.array(sorted(contours[i])).astype(np.int32))

            # contour=-1代表全部画出
            # 0代表不填充，负数代表填充
            cv2.drawContours(mask, contour, -1, 1, -1)

            mask = mask.astype(np.uint8)
            im_floodfill = fill_hole(mask, contour)
            # print(images_array[key])
            # show_single_view(im_floodfill)
            save_image(patient_id, im_floodfill, images_array[key], key)


# save image and mask
def save_image(patient_id: int, mask: np, image: np, key: int):
    """
    save image and mask to 2D image
    :param patient_id:
    :param mask: numpy
    :param image: numpy
    :param key: frame
    :return:
    """
    image_path = os.path.join(config.artery_image_root, f"{patient_id}_{key}.npy")
    np.save(image_path, image)
    mask_path = os.path.join(config.artery_mask_root, f"{patient_id}_{key}.npy")
    np.save(mask_path, mask)


def fill_hole(image, contours):
    """
    图像说明：
    图像为二值化图像，255白色为目标物，0黑色为背景
    要填充白色目标物中的黑色空洞
    """
    im_in = image
    # 复制 im_in 图像
    im_floodfill = im_in.copy()

    # Mask 用于 floodFill，官方要求长宽+2
    h, w = im_in.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # floodFill函数中的seedPoint对应像素必须是背景
    isbreak = False
    # for contour in contours:
    #     for j in contour:
    #         print(j+1,j)

    cv2.floodFill(im_floodfill, mask,(0,0), 1)
    # show_single_view(im_floodfill, cmap="gray")
    # print(im_floodfill)
    # show_single_view(1 - im_floodfill,title="re", cmap="gray")
    # print(1 - im_floodfill.max(),(1 - im_floodfill/im_floodfill.max()).min())
    # 得到im_floodfill的逆im_floodfill_inv
    # im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    reversed_image = 1 - im_floodfill
    # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
    im_out = im_in + reversed_image
    # print(im_out.max())
    return im_out


# 获取动脉
def get_artery():
    # D:\study\dataset\kits21 - master\kits21\data\case_00000\raw\artery\00
    data_path = r"/public/datasets/kidney/kits21/kits21/data"
    delineation_path = r"raw/artery/00/delineation"

    # print(len(os.listdir(data_path)))
    # all patients json files
    for patient, patient_id in zip(os.listdir(data_path), range(len(os.listdir(data_path)))):
        case_path = os.path.join(data_path, patient, delineation_path)
        # print(case_path)
        delineations = list()
        # one patient json files
        is_break = False
        for i in range(1, 2):
            json_path = case_path+str(i)
            if not os.path.exists(json_path):
                delineations.append([])
                is_break = True
                break
            # print(os.listdir(json_path)[0])
            json_file = os.path.join(json_path, os.listdir(json_path)[0])
            # print(patient, json_file)
            delineations.append(read_json(json_file))
            # print(len(delineations[patient_id].get(96)))
        # contour image
        if not is_break:
           image_root = os.path.join(data_path, patient, "imaging.nii.gz")
           show_contours(image_root, delineations, patient_id)


# get kits21 content：tumor，kidney，cyst
def get_kits():
    """
    数据预处理，CT value，slice_size, spacing(save,复原需要用到原来的尺寸)
    将数据分为5折交叉验证：patient%5
    :return:
    """
    # 遍历图片文件aggregated_MAJ_seg.nii.gz
    print("kits")
    data_path = "/public/datasets/kidney/kits21/kits21/data"

    # all patients gz files
    for patient, patient_id in zip(os.listdir(data_path), range(len(os.listdir(data_path)))):
        # image_k_fold = os.path.join(config.kits_image_root, str(patient_id % 5))
        # mask_k_fold = os.path.join(config.kits_mask_root, str(patient_id % 5))
        #
        # if not os.path.exists(image_k_fold):
        #     os.makedirs(image_k_fold)
        # if not os.path.exists(mask_k_fold):
        #     os.makedirs(mask_k_fold)
        image_case_path = os.path.join(data_path, patient, "imaging.nii.gz")
        mask_case_path = os.path.join(data_path, patient, "aggregated_MAJ_seg.nii.gz")
        # print(mask_case_path)
        images = sitk.ReadImage(image_case_path)
        masks = sitk.ReadImage(mask_case_path)
        # # resample
        #
        # resample_image = resize_image_itk(images)
        # resample_mask = resize_image_itk(masks)
        # resample_image = sitk.GetArrayFromImage(resample_image)
        # resample_mask = sitk.GetArrayFromImage(resample_mask)
        # nii为存储路径
        image_save_path = os.path.join(config.kits_image_root, patient+".nii.gz")
        mask_save_path = os.path.join(config.kits_mask_root, patient+".nii.gz")
        print(image_case_path)

        sitk.WriteImage(images, image_save_path)
        sitk.WriteImage(masks, mask_save_path)


def test():
    import shutil
    srcdir = "/public/datasets/kidney/kits21/kits21/data/"
    dstdir = "/public/home/cxiao/Study/data/kits21/val/image"
    image_path = sorted(os.listdir(srcdir))[-30:]
    print(image_path)
    for file_name in image_path:
        srcfile = os.path.join(srcdir, file_name, "imaging.nii.gz")
        dstfile = os.path.join(dstdir, file_name+f".nii.gz")
        shutil.copyfile(srcfile, dstfile)


# kits 2d
def get_kits_2d(begin_case=0, final_case=270):
    data_path = "/public/datasets/kidney/kits21/kits21/data"

    # all patients gz files
    images_path = sorted(os.listdir(data_path))[begin_case:final_case]
    for patient, patient_id in zip(images_path, range(begin_case, final_case)):
        image_fold = os.path.join(config.kits_image_root)
        mask_fold = os.path.join(config.kits_mask_root)

        if not os.path.exists(image_fold):
            os.makedirs(mask_fold)
        if not os.path.exists(image_fold):
            os.makedirs(mask_fold)
        image_case_path = os.path.join(data_path, patient, "imaging.nii.gz")
        mask_case_path = os.path.join(data_path, patient, "aggregated_MAJ_seg.nii.gz")
      #     images_path.append(image_case_path)
        # print(sorted(images_path)[])
        # print(mask_case_path)
        images = sitk.ReadImage(image_case_path)
        masks = sitk.ReadImage(mask_case_path)
        # resample

        images = sitk.GetArrayFromImage(images).transpose((2, 1, 0))
        masks = sitk.GetArrayFromImage(masks).transpose((2, 1, 0))
        print(images.shape, patient)
        if images.shape[1] != images.shape[2]:
            print(images.shape, patient)
        # nonzero = np.nonzero(masks)[0]
        # max_index = nonzero.max()
        # min_index = nonzero.min()
        # nonzero_mask = masks[min_index:max_index, :, :]
        # nonzero_image = images[min_index:max_index, :, :]
        # print(images.shape)
        # nii为存储路径ss
        for index in range(len(masks)):
            image_save_path = os.path.join(image_fold, f"{patient}_{index}.npy")
            mask_save_path = os.path.join(mask_fold, f"{patient}_{index}.npy")
            np.save(mask_save_path, masks[index])
            np.save(image_save_path, images[index])
            # exit(-1)


def calculate_classes(dataset):
    one = 1
    two = 2
    three = 3
    counter = {"1": 0, "2":0, "3":0}
    for i in range(len(dataset)):
        mask = np.array(dataset[i]["mask"].squeeze(1))
        print(mask.shape)
        for j in range(len(mask)):
            if one in mask[j]:
                counter["1"] += 1
            if two in mask[j]:
                counter["2"] += 1
            if three in mask[j]:
                counter["3"] += 1

    return counter


def calculate_all():
    data_path = "/public/datasets/kidney/kits21/kits21/data"

    # all patients gz files
    images_path = sorted(os.listdir(data_path))
    one = 1
    two = 2
    three = 3
    counter = {"1": 0, "2": 0, "3": 0,"all":0}
    case = {"1": 0, "2": 0, "3": 0}
    tumor = list()
    for patient, patient_id in zip(images_path, range(0, 300)):
        image_fold = os.path.join(config.kits_image_root)
        mask_fold = os.path.join(config.kits_mask_root)

        if not os.path.exists(image_fold):
            os.makedirs(mask_fold)
        if not os.path.exists(image_fold):
            os.makedirs(mask_fold)
        image_case_path = os.path.join(data_path, patient, "imaging.nii.gz")
        mask_case_path = os.path.join(data_path, patient, "aggregated_MAJ_seg.nii.gz")
      #     images_path.append(image_case_path)
        # print(sorted(images_path)[])
        # print(mask_case_path)
        images = sitk.ReadImage(image_case_path)
        masks = sitk.ReadImage(mask_case_path)
        # resample

        images = sitk.GetArrayFromImage(images).transpose((2, 1, 0))
        masks = sitk.GetArrayFromImage(masks).transpose((2, 1, 0))
        if images.shape[1] != images.shape[2]:
            print(images.shape, patient)
            continue
        nonzero = np.nonzero(masks)[0]
        max_index = nonzero.max()
        min_index = nonzero.min()
        nonzero_mask = np.array(masks[min_index:max_index, :, :])
        counter["all"] += len(nonzero_mask)
        # print(nonzero_mask.shape)
        for j in range(len(nonzero_mask)):
            if one in nonzero_mask[j]:
                counter["1"] += 1
            if two in nonzero_mask[j]:
                counter["2"] += 1
            if three in nonzero_mask[j]:
                counter["3"] += 1
        if counter["1"] > 0:
            case["1"] += 1
        if counter["2"] > 0:
            case["2"] += 1
        if counter["3"] > 0:
            case["3"] += 1
            tumor.append(patient_id)

        return counter, case, tumor


class Metrics_thread(Thread):
    def __init__(self, file_path, begin_case, final_case):
        Thread.__init__(self)
        self.result = 0
        self.begin_case = begin_case
        self.final_case = final_case
        self.file_path = file_path

    def run(self):
        # print("start")
        self.result = get_cases_spacing(self.file_path, self.begin_case, self.final_case)

    def result(self):
        return self.result


def get_cases_spacing(file_path, start=0, end=300, target_type=".nii.gz"):
    """

    :param file_path: case or cube path
    :param target_type: ".nii.gz"
    :param start:
    :param end:
    :return: dict {"filename":spacing}
    """
    cases_spacing = dict()

    for case_name in os.listdir(file_path)[start:end]:
        case_path = os.path.join(file_path, case_name)
        print(case_name)
        if case_path.endswith(".nii.gz"):
            _, spacing = Convert.nii_2_np(case_path)
            cases_spacing[case_name.split(".")[0]] = spacing.tolist()

    return cases_spacing


def calculate_distribute(lst, intervals=5):
    dic = dict()
    for k, g in groupby(lst, key=lambda x: (x - 1) // intervals):
        dic['{}-{}'.format(k * 10 + 1, (k + 1) * intervals)] = len(list(g))
    return dic


def copy_cases(begin_case=0, final_case=270):
    print("kits")
    data_path = "/public/datasets/kidney/kits21/kits21/data"

    # all patients gz files
    for patient, patient_id in zip(os.listdir(data_path), range(len(os.listdir(data_path)))):
        # image_k_fold = os.path.join(config.kits_image_root, str(patient_id % 5))
        # mask_k_fold = os.path.join(config.kits_mask_root, str(patient_id % 5))
        #
        # if not os.path.exists(image_k_fold):
        #     os.makedirs(image_k_fold)
        # if not os.path.exists(mask_k_fold):
        #     os.makedirs(mask_k_fold)
        image_case_path = os.path.join(data_path, patient, "imaging.nii.gz")
        mask_case_path = os.path.join(data_path, patient, "aggregated_MAJ_seg.nii.gz")
        image_save_path = os.path.join(config.image_3d_path, patient+".nii.gz")
        mask_save_path = os.path.join(config.mask_3d_path, patient+".nii.gz")

        os.system(f'cp {image_case_path} {image_save_path}')
        os.system(f'cp {mask_case_path} {mask_save_path}')
        print(image_save_path)


def get_case_name():
    # print("kits")
    data_path = "/public/datasets/kidney/kits21/kits21/data"

    # all patients gz files
    cases_list = list()
    for patient in os.listdir(config.mask_3d_path):
        if patient.endswith(".nii.gz"):
            patient_id = patient.split(".")[0]
            # print(patient_id)
            cases_list.append(patient_id)
    return {"cases_name": cases_list}


def get_cube_name():
    """
    get cube file name
    :return: dict
    """
    # print("kits")
    data_path = config.crop_cube_image

    # all patients gz files
    cases_list = list()
    for patient in os.listdir(data_path):
        if patient.endswith(".nii.gz"):
            patient_id = patient.split(".")[0]
            # print(patient_id)
            cases_list.append(patient_id)
    return {"cases_name": cases_list}


def get_all_kits():
    case_thread = list()
    result = dict()
    for i in range(10):
        case_thread.append(Metrics_thread(config.image_3d_path, i*30, 30*(i+1)))
        case_thread[i].start()
    for i in range(10):
        case_thread[i].join()
        result.update(case_thread[i].result())
        print("end")
    return result


def data_crop(img, crop_xy):
    """crop data xy to target size

    Args:
        img : image, [c,z,x,y] or [c, x, y]
        crop_xy : target size, [x, y]

    Returns:
        img_crop
    """
    img_shape = img.shape

    if len(img_shape) == 3:
        if type(img) == np.ndarray:
            img = np.expand_dims(img, 0)
        else:
            img = torch.unsqueeze(img, dim=0)

    center = (int(img_shape[-2] / 2), int(img_shape[-1] / 2))

    x_edge = [center[0] - int(crop_xy[0] / 2), center[0] + int(crop_xy[0] / 2)]
    y_edge = [center[1] - int(crop_xy[1] / 2), center[1] + int(crop_xy[1] / 2)]

    img = img[:, :, x_edge[0]:x_edge[1], y_edge[0]:y_edge[1]]

    if len(img_shape) == 3:
        if type(img) == np.ndarray:
            img = np.squeeze(img, 0)
        else:
            img = torch.squeeze(img, dim=0)

    return img


def z_resample(img, spacing, re_thickness=3.5, re_xy=[512, 512], interpolation=cv2.INTER_NEAREST):
    """resample img to target thickness and target xy size (64, 256, 256)[z:1, x:1, y:1]-> (32, 512, 512)[z:2, x:0.5, y:0.5]

    Args:
        img : image, [z, x, y]
        spacing : spacing, [x_spacing, y_spacing, z_spacing]
        re_thickness (int, optional): target z thickness. Defaults to 2.
        re_xy (list, optional): target xy size. Defaults to [512, 512].
        interpolation ([type], optional): interpolation method. Defaults to cv2.INTER_NEAREST.

    Returns:
        resample_image, new_spacing
    """

    # RESCALE ALONG Z-AXIS.
    resize_x = 1.0
    # print(spacing[0], re_thickness)
    resize_z = float(spacing[0]) / float(re_thickness)
    # print(img.shape, spacing[0], resize_z)
    img = cv2.resize(img, dsize=None, fx=resize_x, fy=resize_z, interpolation=interpolation)

    # CHANGE ORDER: (z, x, y) -> (x, y, z).
    img = img.swapaxes(0, 2)
    img = img.swapaxes(0, 1)
    # print(img.shape)
    # RESCALE ALONG XY-AXIS.
    resize_x = float(re_xy[0] / float(img.shape[0]))
    resize_y = float(re_xy[1] / float(img.shape[1]))
    img = cv2.resize(img, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)

    # CHANGE ORDER: (x, y, z) -> (z, x, y).
    # print(img.shape)
    img = img.swapaxes(0, 2)
    # print(img.shape)
    img = img.swapaxes(2, 1)

    new_spacing = [re_thickness, spacing[1] / resize_x, spacing[2] / resize_y]

    return img, new_spacing


def data_windowing(img,  windowing):
    """windowing data

    Args:
        img : image
        windowing: [bottom, top]

    Returns:
        image_windowing
    """
    img[img < windowing[0]] = windowing[0]
    img[img > windowing[1]] = windowing[1]
    img = (img - 101) / 76.9
    return img


def preprocess(config, img_arr, old_spacing):
    """ data preprocess

    Args:
        img_arr (np): image array
        mask_arr (np): mask array
        ct_info (list): spacing [z,x,y]

    Returns:
        [type]: processed image, processed mask, new ct information, old_spacing
    """
    # resample
    img_arr, new_spacing = z_resample(img_arr, old_spacing, re_thickness=config.thickness, re_xy=config.resize_xy)

    # Windowing
    img_arr = data_windowing(img_arr, config.windowing)
    # mask_arr = data_windowing(mask_arr, self.windowing)

    # z pading
    if img_arr.shape[0] < config.layer_thick:
        p = config.layer_thick - img_arr.shape[0]
        img_arr = np.pad(img_arr, ((0, p), (0, 0), (0, 0)), 'minimum')
    # print('pading shape', img_arr.shape, mask_arr.shape, self.layer_thick)
    assert img_arr.shape[0] >= config.layer_thick

    # change spacing
    new_spacing = tuple(new_spacing)

    return img_arr, new_spacing


def case_spacing_distribute():
    cases_space_json = pd.read_json(config.cases_spacing_json)
    print(cases_space_json["case_00049"])
    case_id = [0 for i in range(300)]
    for case in cases_space_json:
        id = int(case[7:10])
        case_id[id] = cases_space_json[case][0]
    plt.hist(np.array(case_id), 300)
    plt.show()


if __name__ == "__main__":
    case_spacing_distribute()

    # image = config.image_3d_path + "/case_00223.nii.gz"
    # mask = config.mask_3d_path + "/case_00223.nii.gz"
    # result = get_cube_name()
    # print(result)
    # DataPrepare3D.info_to_json(config.cube_name_json, result)
    # # result = get_cases_spacing(config.mask_3d_path)
    # result = get_case_name()
    # DataPrepare3D.info_to_json(config.case_name_json, result)
    # # get_all_kits()

