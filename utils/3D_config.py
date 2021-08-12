'''
Author: your name
Date: 2021-08-11 09:37:42
LastEditTime: 2021-08-11 09:37:43
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /kidney_ai/config.py
'''
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# here put the import lib
import os
from easydict import EasyDict as edict

__C = edict()
cfg = __C  # we import cfg in other file  (e.g. from utils.config import cfg)

# Data
__C.DATA = edict()
__C.DATA.root_path = os.path.join('/public', 'kidney', 'pzsz_150')
__C.DATA.crop_dataset_path = os.path.join(__C.DATA.root_path, 'crop_dataset_cs')
__C.DATA.data_detail_json = os.path.join(__C.DATA.root_path, 'data_detail_cs.json')
__C.DATA.data_classify_json = os.path.join(__C.DATA.root_path, 'data_classify_cs.json')

__C.DATA.proc_data = os.path.join(__C.DATA.root_path, 'processed_data_3d_cs')
__C.DATA.imgs_2d_train = os.path.join(__C.DATA.proc_data, 'imgs_train')
__C.DATA.mask_2d_train = os.path.join(__C.DATA.proc_data, 'masks_train')
__C.DATA.imgs_2d_valid = os.path.join(__C.DATA.proc_data, 'imgs_valid')
__C.DATA.mask_2d_valid = os.path.join(__C.DATA.proc_data, 'masks_valid')

__C.DATA.case_imgs_name = 'px_image.nii.gz'
__C.DATA.case_mask_name = 'px_mask.nii.gz'

# CT
__C.CT = edict()
__C.CT.windowing = [-135, 215]

__C.CT.thickness = 1.25

# enh
__C.DATA.trans_types = [
    'AffineNoTranslate', 'AffineNoTranslate', 'HorizontalFlip', 'HorizontalFlip_Affine', 'HorizontalFlip_Affine'
]

__C.DATA.classes_name = ['background', 'collection_system']
__C.DATA.channels_last = False
__C.DATA.layer_thick = 32
__C.DATA.stride = 8
__C.DATA.resize_xy = (512, 512)
__C.DATA.area_size = (__C.DATA.layer_thick, 384, 384)
__C.DATA.crop_xy = (__C.DATA.area_size[1], __C.DATA.area_size[2])
__C.DATA.change_lable = {'2': 0, '3': 0}

# Net
__C.NET = edict()
__C.NET.input_channel = 1
__C.NET.classes = len(__C.DATA.classes_name)
__C.NET.encoder = 'res_unet_3d'
__C.NET.encoder_weights = None
__C.NET.model_name = 'res_unet3d_cs'

# Training
__C.TRAINING = edict()
__C.TRAINING.model_root_path = os.path.join('/public', 'kidney', 'model_test')

__C.TRAINING.epochs = 200
__C.TRAINING.batch_size = 2
__C.TRAINING.val_batch_size = 2
__C.TRAINING.lr = 0.01
__C.TRAINING.weight_decay = 1e-8
__C.TRAINING.adam_betas = [0.9, 0.999]
__C.TRAINING.weight_path = os.path.join('C:\\', 'cz', 'Project', 'Data', 'Kidney', 'model_',
                                        '')  # the path of model weight we want to save
__C.TRAINING.is_valid = True
__C.TRAINING.k_fold = 5
__C.TRAINING.epoch_save = 2

# Test
__C.TEST = edict()
__C.TEST.model_path = os.path.join('/public', 'home', 'Dio', 'model_2', 'unet_3d_stone-2021-07-14-22-54')
__C.TEST.model_name = ''
__C.TEST.output_name = 'results.json'
__C.TEST.test_results_path = ''

# prediction
__C.PRED = edict()
__C.PRED.model_path = os.path.join('/public', 'kidney', 'model_test')
__C.PRED.model_dic = {'croseg_model_name': 'model1', 'reseg_model_name': 'model2'}
__C.PRED.output_path = os.path.join('/public', 'kidney', 'model_test')
